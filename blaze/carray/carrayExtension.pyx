#########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted -  francesc@continuum.io
#
########################################################################


import sys
import numpy as np
import blaze.carray as ca
from blaze.carray import utils, attrs, array2string
import os, os.path
import struct
import shutil
import tempfile
import json
import cython


_KB = 1024
_MB = 1024*_KB

# Directories for saving the data and metadata for carray persistency
DATA_DIR = 'data'
META_DIR = 'meta'
SIZES_FILE = 'sizes'
STORAGE_FILE = 'storage'

# For the persistence layer
EXTENSION = '.blp'
MAGIC = 'blpk'
BLOSCPACK_HEADER_LENGTH = 16
BLOSC_HEADER_LENGTH = 16
FORMAT_VERSION = 1
MAX_FORMAT_VERSION = 255
MAX_CHUNKS = (2**63)-1

# The type used for size values: indexes, coordinates, dimension
# lengths, row numbers, shapes, chunk shapes, byte counts...
SizeType = np.int64

# The native int type for this platform
IntType = np.dtype(np.int_)

#-----------------------------------------------------------------

# numpy functions & objects
from definitions cimport import_array, ndarray, dtype, \
     malloc, realloc, free, memcpy, memset, strdup, strcmp, \
     PyString_AsString, PyString_GET_SIZE, PyString_FromString, \
     PyString_FromStringAndSize, \
     Py_BEGIN_ALLOW_THREADS, Py_END_ALLOW_THREADS, \
     PyArray_GETITEM, PyArray_SETITEM, \
     npy_intp, PyBuffer_FromMemory, Py_uintptr_t

#-----------------------------------------------------------------


# Blosc routines
cdef extern from "blosc.h":

  cdef enum:
    BLOSC_MAX_OVERHEAD,
    BLOSC_VERSION_STRING,
    BLOSC_VERSION_DATE

  void blosc_get_versions(char *version_str, char *version_date)
  int blosc_set_nthreads(int nthreads)
  int blosc_compress(int clevel, int doshuffle, size_t typesize,
                     size_t nbytes, void *src, void *dest,
                     size_t destsize) nogil
  int blosc_decompress(void *src, void *dest, size_t destsize) nogil
  int blosc_getitem(void *src, int start, int nitems, void *dest) nogil
  void blosc_free_resources()
  void blosc_cbuffer_sizes(void *cbuffer, size_t *nbytes,
                           size_t *cbytes, size_t *blocksize)
  void blosc_cbuffer_metainfo(void *cbuffer, size_t *typesize, int *flags)
  void blosc_cbuffer_versions(void *cbuffer, int *version, int *versionlz)
  void blosc_set_blocksize(size_t blocksize)


#----------------------------------------------------------------------------

# Initialization code

# The numpy API requires this function to be called before
# using any numpy facilities in an extension module.
import_array()

#-------------------------------------------------------------

# Some utilities
def _blosc_set_nthreads(nthreads):
  """
  blosc_set_nthreads(nthreads)

  Sets the number of threads that Blosc can use.

  Parameters
  ----------
  nthreads : int
      The desired number of threads to use.

  Returns
  -------
  out : int
      The previous setting for the number of threads.

  """
  return blosc_set_nthreads(nthreads)

def blosc_version():
  """
  blosc_version()

  Return the version of the Blosc library.

  """
  return (<char *>BLOSC_VERSION_STRING, <char *>BLOSC_VERSION_DATE)

# This is the same than in utils.py, but works faster in extensions
cdef get_len_of_range(npy_intp start, npy_intp stop, npy_intp step):
  """Get the length of a (start, stop, step) range."""
  cdef npy_intp n

  n = 0
  if start < stop:
    # Do not use a cython.cdiv here (do not ask me why!)
    n = ((stop - start - 1) // step + 1)
  return n

cdef clip_chunk(npy_intp nchunk, npy_intp chunklen,
                npy_intp start, npy_intp stop, npy_intp step):
  """Get the limits of a certain chunk based on its length."""
  cdef npy_intp startb, stopb, blen, distance

  startb = start - nchunk * chunklen
  stopb = stop - nchunk * chunklen

  # Check limits
  if (startb >= chunklen) or (stopb <= 0):
    return startb, stopb, 0   # null size
  if startb < 0:
    startb = 0
  if stopb > chunklen:
    stopb = chunklen

  # step corrections
  if step > 1:
    # Just correcting startb is enough
    distance = (nchunk * chunklen + startb) - start
    if distance % step > 0:
      startb += (step - (distance % step))
      if startb > chunklen:
        return startb, stopb, 0  # null size

  # Compute size of the clipped block
  blen = get_len_of_range(startb, stopb, step)

  return startb, stopb, blen

cdef int check_zeros(char *data, int nbytes):
  """Check whether [data, data+nbytes] is zero or not."""
  cdef int i, iszero, chunklen, leftover
  cdef size_t *sdata

  iszero = 1
  sdata = <size_t *>data
  chunklen = cython.cdiv(nbytes, sizeof(size_t))
  leftover = nbytes % sizeof(size_t)
  with nogil:
    for i from 0 <= i < chunklen:
      if sdata[i] != 0:
        iszero = 0
        break
    else:
      data += nbytes - leftover
      for i from 0 <= i < leftover:
        if data[i] != 0:
          iszero = 0
          break
  return iszero

cdef int true_count(char *data, int nbytes):
  """Count the number of true values in data (boolean)."""
  cdef int i, count

  with nogil:
    count = 0
    for i from 0 <= i < nbytes:
      count += <int>(data[i])
  return count

#-------------------------------------------------------------

# For member defintions see carrayExtension.pxd ~Stephen
cdef class chunk:
  """
  chunk(array, atom, cparams)

  Compressed in-memory container for a data chunk.

  This class is meant to be used only by the `carray` class.

  """
  cdef char typekind, isconstant
  cdef public int atomsize, itemsize, blocksize
  cdef public int nbytes, cbytes, cdbytes
  cdef int true_count
  cdef char *data
  cdef object atom, constant, dobject

  cdef void _getitem(self, int start, int stop, char *dest)
  cdef compress_data(self, char *data, size_t itemsize, size_t nbytes, object cparams)
  cdef compress_arrdata(self, ndarray array, object cparams, object _memory)

  property dtype:
    "The NumPy dtype for this chunk."
    def __get__(self):
      return self.atom

  def __cinit__(self, object dobject, object atom, object cparams,
                object _memory=True, object _compr=False):
    cdef int itemsize, footprint
    cdef size_t nbytes, cbytes, blocksize
    cdef dtype dtype_
    cdef char *data

    self.atom = atom
    self.atomsize = atom.itemsize
    dtype_ = atom.base
    self.itemsize = itemsize = dtype_.elsize
    self.typekind = dtype_.kind
    self.dobject = None
    footprint = 0

    if _compr:
      # Data comes in an already compressed state inside a Python String
      self.data = PyString_AsString(dobject)
      # Increment the reference so that data don't go away
      self.dobject = dobject 
      # Set size info for the instance
      blosc_cbuffer_sizes(self.data, &nbytes, &cbytes, &blocksize)
    elif dtype_ == 'O':
      # The objects should arrive here already pickled
      data = PyString_AsString(dobject)
      nbytes = PyString_GET_SIZE(dobject)
      cbytes, blocksize = self.compress_data(data, 1, nbytes, cparams)
    else:
      # Compress the data object (a NumPy object)
      nbytes, cbytes, blocksize, footprint = self.compress_arrdata(
        dobject, cparams, _memory)
    footprint += 128  # add the (aprox) footprint of this instance in bytes

    # Fill instance data
    self.nbytes = nbytes
    self.cbytes = cbytes + footprint
    self.cdbytes = cbytes
    self.blocksize = blocksize

  cdef compress_arrdata(self, ndarray array, object cparams, object _memory):
    """Compress data in `array` and put it in ``self.data``"""
    cdef size_t nbytes, cbytes, blocksize, itemsize, footprint

    # Compute the total number of bytes in this array
    itemsize = array.itemsize
    nbytes = itemsize * array.size
    cbytes = 0
    footprint = 0

    # Check whether incoming data can be expressed as a constant or not.
    # Disk-based chunks are not allowed to do this.
    self.isconstant = 0
    self.constant = None
    if _memory and (array.strides[0] == 0
                    or check_zeros(array.data, nbytes)):

      self.isconstant = 1
      # Get the NumPy constant.  Avoid this NumPy quirk:
      # np.array(['1'], dtype='S3').dtype != s[0].dtype
      if array.dtype.kind != 'S':
        self.constant = array[0]
      else:
        self.constant = np.array(array[0], dtype=array.dtype)
      # Add overhead (64 bytes for the overhead of the numpy container)
      footprint += 64 + self.constant.size * self.constant.itemsize

    if self.isconstant:
      blocksize = 4*1024  # use 4 KB as a cache for blocks
      # Make blocksize a multiple of itemsize
      if blocksize % itemsize > 0:
        blocksize = cython.cdiv(blocksize, itemsize) * itemsize
      # Correct in case we have a large itemsize
      if blocksize == 0:
        blocksize = itemsize
    else:
      if self.typekind == 'b':
        self.true_count = true_count(array.data, nbytes)

      if array.strides[0] == 0:
        # The chunk is made of constants.  Regenerate the actual data.
        array = array.copy()

      # Compress data
      cbytes, blocksize = self.compress_data(array.data, itemsize, nbytes,
                                             cparams)

    return (nbytes, cbytes, blocksize, footprint)

  cdef compress_data(self, char *data, size_t itemsize, size_t nbytes,
                     object cparams):
    """Compress data with `caparms` and return metadata."""
    cdef size_t nbytes_, cbytes, blocksize
    cdef int clevel, shuffle
    cdef char *dest

    clevel = cparams.clevel
    shuffle = cparams.shuffle
    dest = <char *>malloc(nbytes+BLOSC_MAX_OVERHEAD)
    with nogil:
      cbytes = blosc_compress(clevel, shuffle, itemsize, nbytes,
                              data, dest, nbytes+BLOSC_MAX_OVERHEAD)
    if cbytes <= 0:
      raise RuntimeError, "fatal error during Blosc compression: %d" % cbytes
    # Free the unused data
    self.data = <char *>realloc(dest, cbytes)
    # Set size info for the instance
    blosc_cbuffer_sizes(self.data, &nbytes_, &cbytes, &blocksize)
    assert nbytes_ == nbytes

    return (cbytes, blocksize)

  def getdata(self):
    """Get a compressed string object out of this chunk (for persistence)."""
    cdef object string

    assert (not self.isconstant,
            "This function can only be used for persistency")
    string = PyString_FromStringAndSize(self.data, <Py_ssize_t>self.cdbytes)
    return string

  def getudata(self):
    """Get an uncompressed string out of this chunk (for 'O'bject types)."""
    cdef int ret
    cdef char *dest

    dest = <char *>malloc(self.nbytes)
    # Fill dest with uncompressed data
    with nogil:
      ret = blosc_decompress(self.data, dest, self.nbytes)
    if ret < 0:
      raise RuntimeError, "fatal error during Blosc decompression: %d" % ret
    string = PyString_FromStringAndSize(dest, <Py_ssize_t>self.nbytes)
    return string

  cdef void _getitem(self, int start, int stop, char *dest):
    """Read data from `start` to `stop` and return it as a numpy array."""
    cdef int ret, bsize, blen, nitems, nstart
    cdef ndarray constants

    blen = stop - start
    bsize = blen * self.atomsize
    nitems = cython.cdiv(bsize, self.itemsize)
    nstart = cython.cdiv(start * self.atomsize, self.itemsize)

    if self.isconstant:
      # The chunk is made of constants
      constants = np.ndarray(shape=(blen,), dtype=self.dtype,
                             buffer=self.constant, strides=(0,)).copy()
      memcpy(dest, constants.data, bsize)
      return

    # Fill dest with uncompressed data
    with nogil:
      if bsize == self.nbytes:
        ret = blosc_decompress(self.data, dest, bsize)
      else:
        ret = blosc_getitem(self.data, nstart, nitems, dest)
    if ret < 0:
      raise RuntimeError, "fatal error during Blosc decompression: %d" % ret

  def __getitem__(self, object key):
    """__getitem__(self, key) -> values."""
    cdef ndarray array
    cdef object start, stop, step, clen, idx

    if isinstance(key, (int, long)):
      # Quickly return a single element
      array = np.empty(shape=(1,), dtype=self.dtype)
      self._getitem(key, key+1, array.data)
      return PyArray_GETITEM(array, array.data)
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
    elif isinstance(key, tuple) and self.dtype.shape != ():
      # Build an array to guess indices
      clen = cython.cdiv(self.nbytes, self.itemsize)
      idx = np.arange(clen, dtype=np.int32).reshape(self.dtype.shape)
      idx2 = idx(key)
      if idx2.flags.contiguous:
        # The slice represents a contiguous slice.  Get start and stop.
        start, stop = idx2.flatten()[[0,-1]]
        step = 1
      else:
        (start, stop, step) = key[0].start, key[0].stop, key[0].step
    else:
      raise IndexError, "key not suitable:", key

    # Get the corrected values for start, stop, step
    clen = cython.cdiv(self.nbytes, self.atomsize)
    (start, stop, step) = slice(start, stop, step).indices(clen)

    # Build a numpy container
    array = np.empty(shape=(stop-start,), dtype=self.dtype)
    # Read actual data
    self._getitem(start, stop, array.data)

    # Return the value depending on the step
    if step > 1:
      return array[::step]
    return array

  @property
  def pointer(self):
      if self.memory:
          return <Py_uintptr_t> self.data+BLOSCPACK_HEADER_LENGTH
      else:
          raise RuntimeError("Not in memory")

  @property
  def viewof(self):
      if self.memory:
          return PyBuffer_FromMemory(<void*>self.data, <Py_ssize_t>self.cdbytes)
      else:
          raise RuntimeError("Not in memory")


  def __setitem__(self, object key, object value):
    """__setitem__(self, key, value) -> None."""
    raise NotImplementedError

  def __str__(self):
    """Represent the chunk as an string."""
    return str(self[:])

  def __repr__(self):
    """Represent the chunk as an string, with additional info."""
    cratio = self.nbytes / float(self.cbytes)
    fullrepr = "chunk(%s, %s)  nbytes: %d; cbytes: %d; ratio: %.2f\n%r" % \
        (self.shape, self.dtype, self.nbytes, self.cbytes, cratio, str(self))
    return fullrepr

  def __dealloc__(self):
    """Release C resources before destruction."""
    if self.dobject:
      self.dobject = None  # DECREF pointer to data object
    else:
      free(self.data)   # explictly free the data area


cdef create_bloscpack_header(nchunks=None, format_version=FORMAT_VERSION):
    """ Create the bloscpack header string.

    Parameters
    ----------
    nchunks : int
        the number of chunks, default: None
    format_version : int
        the version format for the compressed file

    Returns
    -------
    bloscpack_header : string
        the header as string

    Notes
    -----

    The bloscpack header is 16 bytes as follows:

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
    | b   l   p   k | ^ | RESERVED  |           nchunks             |
                   version

    The first four are the magic string 'blpk'. The next one is an 8 bit
    unsigned little-endian integer that encodes the format version. The next
    three are reserved, and the last eight are a signed  64 bit little endian
    integer that encodes the number of chunks

    The value of '-1' for 'nchunks' designates an unknown size and can be
    inserted by setting 'nchunks' to None.

    Raises
    ------
    ValueError
        if the nchunks argument is too large or negative
    struct.error
        if the format_version is too large or negative

    """
    if not 0 <= nchunks <= MAX_CHUNKS and nchunks is not None:
      raise ValueError(
        "'nchunks' must be in the range 0 <= n <= %d, not '%s'" %
        (MAX_CHUNKS, str(nchunks)))
    return (MAGIC + struct.pack('<B', format_version) + '\x00\x00\x00' +
            struct.pack('<q', nchunks if nchunks is not None else -1))

def decode_byte(byte):
  return int(byte.encode('hex'), 16)
def decode_uint32(fourbyte):
  return struct.unpack('<I', fourbyte)[0]

cdef decode_blosc_header(buffer_):
    """ Read and decode header from compressed Blosc buffer.

    Parameters
    ----------
    buffer_ : string of bytes
        the compressed buffer

    Returns
    -------
    settings : dict
        a dict containing the settings from Blosc

    Notes
    -----

    The Blosc 1.1.3 header is 16 bytes as follows:

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
      ^   ^   ^   ^ |     nbytes    |   blocksize   |    ctbytes    |
      |   |   |   |
      |   |   |   +--typesize
      |   |   +------flags
      |   +----------versionlz
      +--------------version

    The first four are simply bytes, the last three are are each unsigned ints
    (uint32) each occupying 4 bytes. The header is always little-endian.
    'ctbytes' is the length of the buffer including header and nbytes is the
    length of the data when uncompressed.

    """
    return {'version': decode_byte(buffer_[0]),
            'versionlz': decode_byte(buffer_[1]),
            'flags': decode_byte(buffer_[2]),
            'typesize': decode_byte(buffer_[3]),
            'nbytes': decode_uint32(buffer_[4:8]),
            'blocksize': decode_uint32(buffer_[8:12]),
            'ctbytes': decode_uint32(buffer_[12:16])}


cdef class chunks(object):
  """Store the different carray chunks in a directory on-disk."""
  cdef object _rootdir, _mode
  cdef object dtype, cparams, lastchunkarr
  cdef object chunk_cached
  cdef npy_intp nchunks, nchunk_cached, len

  property mode:
    "The mode used to create/open the `mode`."
    def __get__(self):
      return self._mode
    def __set__(self, value):
      self._mode = value

  property rootdir:
    "The on-disk directory used for persistency."
    def __get__(self):
      return self._rootdir
    def __set__(self, value):
      self._rootdir = value

  property datadir:
    """The directory for data files."""
    def __get__(self):
      return os.path.join(self.rootdir, DATA_DIR)

  def __cinit__(self, rootdir, metainfo=None, _new=False):
    cdef ndarray lastchunkarr
    cdef void *decompressed, *compressed
    cdef int leftover
    cdef char *lastchunk
    cdef size_t chunksize
    cdef object scomp
    cdef int ret
    cdef int itemsize, atomsize

    self._rootdir = rootdir
    self.nchunks = 0
    self.nchunk_cached = -1    # no chunk cached initially
    self.dtype, self.cparams, self.len, lastchunkarr, self._mode = metainfo
    atomsize = self.dtype.itemsize
    itemsize = self.dtype.base.itemsize

    # For 'O'bject types, the number of chunks is equal to the number of
    # elements
    if self.dtype.char == 'O':
      self.nchunks = self.len

    # Initialize last chunk (not valid for 'O'bject dtypes)
    if not _new and self.dtype.char != 'O':
      self.nchunks = cython.cdiv(self.len, len(lastchunkarr))
      chunksize = len(lastchunkarr) * atomsize
      lastchunk = lastchunkarr.data
      leftover = (self.len % len(lastchunkarr)) * atomsize
      if leftover:
        # Fill lastchunk with data on disk
        scomp = self.read_chunk(self.nchunks)
        compressed = PyString_AsString(scomp)
        with nogil:
          ret = blosc_decompress(compressed, lastchunk, chunksize)
        if ret < 0:
          raise RuntimeError(
            "error decompressing the last chunk (error code: %d)" % ret)

  cdef read_chunk(self, nchunk):
    """Read a chunk and return it in compressed form."""
    dname = "__%d%s" % (nchunk, EXTENSION)
    schunkfile = os.path.join(self.datadir, dname)
    if not os.path.exists(schunkfile):
      raise ValueError("chunkfile %s not found" % schunkfile)
    with open(schunkfile, 'rb') as schunk:
      bloscpack_header = schunk.read(BLOSCPACK_HEADER_LENGTH)
      blosc_header_raw = schunk.read(BLOSC_HEADER_LENGTH)
      blosc_header = decode_blosc_header(blosc_header_raw)
      ctbytes = blosc_header['ctbytes']
      nbytes = blosc_header['nbytes']
      # seek back BLOSC_HEADER_LENGTH bytes in file relative to current
      # position
      schunk.seek(-BLOSC_HEADER_LENGTH, 1)
      scomp = schunk.read(ctbytes)
    return scomp

  def __getitem__(self, nchunk):
    cdef void *decompressed, *compressed

    if nchunk == self.nchunk_cached:
      # Hit!
      return self.chunk_cached
    else:
      scomp = self.read_chunk(nchunk)
      # Data chunk should be compressed already
      chunk_ = chunk(scomp, self.dtype, self.cparams,
                     _memory=False, _compr=True)
      # Fill cache
      self.nchunk_cached = nchunk
      self.chunk_cached = chunk_
    return chunk_

  def __setitem__(self, nchunk, chunk_):
    self._save(nchunk, chunk_)

  def __len__(self):
    return self.nchunks

  def append(self, chunk_):
    """Append an new chunk to the carray."""
    self._save(self.nchunks, chunk_)
    self.nchunks += 1

  cdef _save(self, nchunk, chunk_):
    """Save the `chunk_` as chunk #`nchunk`. """

    if self.mode == "r":
      raise RuntimeError(
        "cannot modify data because mode is '%s'" % self.mode)

    dname = "__%d%s" % (nchunk, EXTENSION)
    schunkfile = os.path.join(self.datadir, dname)
    bloscpack_header = create_bloscpack_header(1)
    with open(schunkfile, 'wb') as schunk:
      schunk.write(bloscpack_header)
      data = chunk_.getdata()
      schunk.write(data)
    # Mark the cache as dirty if needed
    if nchunk == self.nchunk_cached:
      self.nchunk_cached = -1

  def flush(self, chunk_):
    """Flush the leftover chunk."""
    self._save(self.nchunks, chunk_)

  def pop(self):
    """Remove the last chunk and return it."""
    nchunk = self.nchunks - 1
    chunk_ = self.__getitem__(nchunk)
    dname = "__%d%s" % (nchunk, EXTENSION)
    schunkfile = os.path.join(self.datadir, dname)
    if not os.path.exists(schunkfile):
      raise RuntimeError("chunk filename %s does exist" % schunkfile)
    os.remove(schunkfile)

    # When poping a chunk, we must be sure that we don't leave anything
    # behind (i.e. the lastchunk)
    dname = "__%d%s" % (nchunk+1, EXTENSION)
    schunkfile = os.path.join(self.datadir, dname)
    if os.path.exists(schunkfile):
      os.remove(schunkfile)

    self.nchunks -= 1
    return chunk_


cdef class carray:
  """
  carray(array, cparams=None, dtype=None, dflt=None, expectedlen=None, chunklen=None, rootdir=None, mode='a')

  A compressed and enlargeable in-memory data container.

  `carray` exposes a series of methods for dealing with the compressed
  container in a NumPy-like way.

  Parameters
  ----------
  array : a NumPy-like object
      This is taken as the input to create the carray.  It can be any Python
      object that can be converted into a NumPy object.  The data type of
      the resulting carray will be the same as this NumPy object.
  cparams : instance of the `cparams` class, optional
      Parameters to the internal Blosc compressor.
  dtype : NumPy dtype
      Force this `dtype` for the carray (rather than the `array` one).
  dflt : Python or NumPy scalar
      The value to be used when enlarging the carray.  If None, the default is
      filling with zeros.
  expectedlen : int, optional
      A guess on the expected length of this object.  This will serve to
      decide the best `chunklen` used for compression and memory I/O
      purposes.
  chunklen : int, optional
      The number of items that fits into a chunk.  By specifying it you can
      explicitely set the chunk size used for compression and memory I/O.
      Only use it if you know what are you doing.
  rootdir : str, optional
      The directory where all the data and metadata will be stored.  If
      specified, then the carray object will be disk-based (i.e. all chunks
      will live on-disk, not in memory) and persistent (i.e. it can be
      restored in other session, e.g. via the `open()` top-level function).
  mode : str, optional
      The mode that a *persistent* carray should be created/opened.  The
      values can be:

        * 'r' for read-only
        * 'w' for read/write.  During carray creation, the `rootdir` will be
          removed if it exists.  During carray opening, the carray will be
          resized to 0.
        * 'a' for append (possible data inside `rootdir` will not be removed).

  """

  cdef public int itemsize, atomsize
  cdef int _chunksize, _chunklen, leftover
  cdef int nrowsinbuf, _row
  cdef int sss_mode, wheretrue_mode, where_mode
  cdef npy_intp startb, stopb
  cdef npy_intp start, stop, step, nextelement
  cdef npy_intp _nrow, nrowsread
  cdef npy_intp _nbytes, _cbytes
  cdef npy_intp nhits, limit, skip
  cdef npy_intp expectedlen
  cdef char *lastchunk
  cdef object lastchunkarr, where_arr, arr1
  cdef object _cparams, _dflt
  cdef object _dtype
  cdef public object chunks
  cdef object _rootdir, datadir, metadir, _mode
  cdef object _attrs
  cdef ndarray iobuf, where_buf
  # For block cache
  cdef int idxcache
  cdef ndarray blockcache
  cdef char *datacache

  property leftovers:
    def __get__(self):
      # Pointer to the leftovers chunk
      return self.lastchunkarr.ctypes.data

  property nchunks:
    def __get__(self):
      # TODO: do we need to handle the last chunk specially?
      return cython.cdiv(self._nbytes, <npy_intp>self._chunksize)

  property partitions:
    def __get__(self):
      # Return a sequence of tuples indicating the bounds
      # of each of the chunks.
      nchunks = cython.cdiv(self._nbytes, <npy_intp>self._chunksize)
      chunklen = cython.cdiv(self._chunksize, self.atomsize)
      return [(i*chunklen,(i+1)*chunklen) for i in xrange(nchunks)]

  property leftover_array:
      def __get__(self):
          return self.lastchunkarr

  property attrs:
    "The attribute accessor."
    def __get__(self):
      return self._attrs

  property cbytes:
    "The compressed size of this object (in bytes)."
    def __get__(self):
      return self._cbytes

  property chunklen:
    "The chunklen of this object (in rows)."
    def __get__(self):
      return self._chunklen

  property cparams:
    "The compression parameters for this object."
    def __get__(self):
      return self._cparams

  property dflt:
    "The default value of this object."
    def __get__(self):
      return self._dflt

  property dtype:
    "The dtype of this object."
    def __get__(self):
      return self._dtype.base

  property len:
    "The length (leading dimension) of this object."
    def __get__(self):
      if self._dtype.char == 'O':
        return len(self.chunks)
      else:
        # Important to do the cast in order to get a npy_intp result
        return cython.cdiv(self._nbytes, <npy_intp>self.atomsize)

  property mode:
    "The mode used to create/open the `mode`."
    def __get__(self):
      return self._mode
    def __set__(self, value):
      self._mode = value
      self.chunks.mode = value

  property nbytes:
    "The original (uncompressed) size of this object (in bytes)."
    def __get__(self):
      return self._nbytes

  property ndim:
    "The number of dimensions of this object."
    def __get__(self):
      return len(self.shape)

  property shape:
    "The shape of this object."
    def __get__(self):
      return tuple((self.len,) + self._dtype.shape)

  property size:
    "The size of this object."
    def __get__(self):
      return np.prod(self.shape)

  property rootdir:
    "The on-disk directory used for persistency."
    def __get__(self):
      return self._rootdir
    def __set__(self, value):
      if not self.rootdir:
        raise ValueError(
          "cannot modify the rootdir value of an in-memory carray")
      self._rootdir = value
      self.chunks.rootdir = value

  def __cinit__(self, object array=None, object cparams=None,
                object dtype=None, object dflt=None,
                object expectedlen=None, object chunklen=None,
                object rootdir=None, object mode="a"):

    self._rootdir = rootdir
    if mode not in ('r', 'w', 'a'):
      raise ValueError("mode should be 'r', 'w' or 'a'")
    self._mode = mode

    if array is not None:
      self.create_carray(array, cparams, dtype, dflt,
                         expectedlen, chunklen, rootdir, mode)
      _new = True
    elif rootdir is not None:
      meta_info = self.read_meta()
      self.open_carray(*meta_info)
      _new = False
    else:
      raise ValueError("You need at least to pass an array or/and a rootdir")

    # Attach the attrs to this object
    self._attrs = attrs.attrs(self._rootdir, self.mode, _new=_new)

    # Cache a len-1 array for accelerating self[int] case
    self.arr1 = np.empty(shape=(1,), dtype=self._dtype)

    # Sentinels
    self.sss_mode = False
    self.wheretrue_mode = False
    self.where_mode = False
    self.idxcache = -1       # cache not initialized

  def create_carray(self, array, cparams, dtype, dflt,
                    expectedlen, chunklen, rootdir, mode):
    """Create a new array."""
    cdef int itemsize, atomsize, chunksize
    cdef ndarray lastchunkarr
    cdef object array_, _dflt

    # Check defaults for cparams
    if cparams is None:
      cparams = ca.cparams()

    if not isinstance(cparams, ca.cparams):
      raise ValueError, "`cparams` param must be an instance of `cparams` class"

    # Convert input to an appropriate type
    if type(dtype) is str:
        dtype = np.dtype(dtype)
    array_ = utils.to_ndarray(array, dtype)
    if dtype is None:
      if len(array_.shape) == 1:
        self._dtype = dtype = array_.dtype
      else:
        # Multidimensional array.  The atom will have array_.shape[1:] dims.
        # atom dimensions will be stored in `self._dtype`, which is different
        # than `self.dtype` in that `self._dtype` dimensions are borrowed
        # from `self.shape`.  `self.dtype` will always be scalar (NumPy
        # convention).
        self._dtype = dtype = np.dtype((array_.dtype.base, array_.shape[1:]))
    else:
      self._dtype = dtype

    # Check that atom size is less than 2 GB
    if dtype.itemsize >= 2**31:
      raise ValueError, "atomic size is too large (>= 2 GB)"

    self.atomsize = atomsize = dtype.itemsize
    self.itemsize = itemsize = dtype.base.itemsize

    # Check defaults for dflt
    _dflt = np.zeros((), dtype=dtype)
    if dflt is not None:
      if dtype.shape == ():
        _dflt[()] = dflt
      else:
        _dflt[:] = dflt
    self._dflt = _dflt

    # Compute the chunklen/chunksize
    if expectedlen is None:
      # Try a guess
      try:
        expectedlen = len(array_)
      except TypeError:
        raise NotImplementedError(
          "creating carrays from scalar objects not supported")
    try:
      self.expectedlen = expectedlen
    except OverflowError:
      raise OverflowError(
        "The size cannot be larger than 2**31 on 32-bit platforms")
    if chunklen is None:
      # Try a guess
      chunksize = utils.calc_chunksize((expectedlen * atomsize) / float(_MB))
      # Chunksize must be a multiple of atomsize
      chunksize = cython.cdiv(chunksize, atomsize) * atomsize
      # Protection against large itemsizes
      if chunksize < atomsize:
        chunksize = atomsize
    else:
      if not isinstance(chunklen, int) or chunklen < 1:
        raise ValueError, "chunklen must be a positive integer"
      chunksize = chunklen * atomsize
    chunklen = cython.cdiv(chunksize, atomsize)
    self._chunksize = chunksize
    self._chunklen = chunklen

    # Book memory for last chunk (uncompressed)
    # Use np.zeros here because they compress better
    lastchunkarr = np.zeros(dtype=dtype, shape=(chunklen,))
    self.lastchunk = lastchunkarr.data
    self.lastchunkarr = lastchunkarr

    # Create layout for data and metadata
    self._cparams = cparams
    self.chunks = []
    if rootdir is not None:
      self.mkdirs(rootdir, mode)
      metainfo = (dtype, cparams, self.shape[0], lastchunkarr, self._mode)
      self.chunks = chunks(self._rootdir, metainfo=metainfo, _new=True)
      # We can write the metainfo already
      self.write_meta()

    # Finally, fill the chunks
    # Object dtype requires special storage
    if array_.dtype.char == 'O':
      for obj in array_:
        self.store_obj(obj)
    else:
      self.fill_chunks(array_)

    # and flush the data pending...
    self.flush()

  def open_carray(self, shape, cparams, dtype, dflt,
                  expectedlen, cbytes, chunklen):
    """Open an existing array."""
    cdef ndarray lastchunkarr
    cdef object array_, _dflt
    cdef npy_intp calen

    if len(shape) == 1:
        self._dtype = dtype
    else:
      # Multidimensional array.  The atom will have array_.shape[1:] dims.
      # atom dimensions will be stored in `self._dtype`, which is different
      # than `self.dtype` in that `self._dtype` dimensions are borrowed
      # from `self.shape`.  `self.dtype` will always be scalar (NumPy
      # convention).
      self._dtype = dtype = np.dtype((dtype.base, shape[1:]))

    self._cparams = cparams
    self.atomsize = dtype.itemsize
    self.itemsize = dtype.base.itemsize
    self._chunklen = chunklen
    self._chunksize = chunklen * self.atomsize
    self._dflt = dflt
    self.expectedlen = expectedlen

    # Book memory for last chunk (uncompressed)
    # Use np.zeros here because they compress better
    lastchunkarr = np.zeros(dtype=dtype, shape=(chunklen,))
    self.lastchunk = lastchunkarr.data
    self.lastchunkarr = lastchunkarr

    # Check rootdir hierarchy
    if not os.path.isdir(self._rootdir):
      raise RuntimeError("root directory does not exist")
    self.datadir = os.path.join(self._rootdir, DATA_DIR)
    if not os.path.isdir(self.datadir):
      raise RuntimeError("data directory does not exist")
    self.metadir = os.path.join(self._rootdir, META_DIR)
    if not os.path.isdir(self.metadir):
      raise RuntimeError("meta directory does not exist")

    calen = shape[0]    # the length ot the carray
    # Finally, open data directory
    metainfo = (dtype, cparams, calen, lastchunkarr, self._mode)
    self.chunks = chunks(self._rootdir, metainfo=metainfo, _new=False)

    # Update some counters
    self.leftover = (calen % chunklen) * self.atomsize
    self._cbytes = cbytes
    self._nbytes = calen * self.atomsize

    if self._mode == "w":
      # Remove all entries when mode is 'w'
      self.resize(0)

  def fill_chunks(self, object array_):
    """Fill chunks, either in-memory or on-disk."""
    cdef int leftover, chunklen
    cdef npy_intp i, nchunks
    cdef npy_intp nbytes, cbytes
    cdef chunk chunk_
    cdef ndarray remainder

    # The number of bytes in incoming array
    nbytes = self.itemsize * array_.size
    self._nbytes = nbytes

    # Compress data in chunks
    cbytes = 0
    chunklen = self._chunklen
    nchunks = cython.cdiv(nbytes, <npy_intp>self._chunksize)
    for i from 0 <= i < nchunks:
      assert i*chunklen < array_.size, "i, nchunks: %d, %d" % (i, nchunks)
      chunk_ = chunk(array_[i*chunklen:(i+1)*chunklen],
                     self._dtype, self._cparams,
                     _memory = self._rootdir is None)
      self.chunks.append(chunk_)
      cbytes += chunk_.cbytes
    self.leftover = leftover = nbytes % self._chunksize
    if leftover:
      remainder = array_[nchunks*chunklen:]
      memcpy(self.lastchunk, remainder.data, leftover)
    cbytes += self._chunksize  # count the space in last chunk
    self._cbytes = cbytes

  def mkdirs(self, object rootdir, object mode):
    """Create the basic directory layout for persistent storage."""
    if os.path.exists(rootdir):
      if self._mode != "w":
        raise RuntimeError(
          "specified rootdir path '%s' already exists "
          "and creation mode is '%s'" % (rootdir, mode))
      if os.path.isdir(rootdir):
        shutil.rmtree(rootdir)
      else:
        os.remove(rootdir)
    os.mkdir(rootdir)
    self.datadir = os.path.join(rootdir, DATA_DIR)
    os.mkdir(self.datadir)
    self.metadir = os.path.join(rootdir, META_DIR)
    os.mkdir(self.metadir)

  def write_meta(self):
      """Write metadata persistently."""
      storagef = os.path.join(self.metadir, STORAGE_FILE)
      with open(storagef, 'wb') as storagefh:
        storagefh.write(json.dumps({
          "dtype": str(self.dtype),
          "cparams": {
            "clevel": self.cparams.clevel,
            "shuffle": self.cparams.shuffle,
            },
          "chunklen": self._chunklen,
          "expectedlen": self.expectedlen,
          "dflt": self.dflt.tolist(),
          }))
        storagefh.write("\n")

  def read_meta(self):
    """Read persistent metadata."""

    # First read the size info
    metadir = os.path.join(self._rootdir, META_DIR)
    shapef = os.path.join(metadir, SIZES_FILE)
    with open(shapef, 'rb') as shapefh:
      sizes = json.loads(shapefh.read())
    shape = sizes['shape']
    if type(shape) == list:
      shape = tuple(shape)
    nbytes = sizes["nbytes"]
    cbytes = sizes["cbytes"]

    # Then the rest of metadata
    storagef = os.path.join(metadir, STORAGE_FILE)
    with open(storagef, 'rb') as storagefh:
      data = json.loads(storagefh.read())
    dtype_ = np.dtype(data["dtype"])
    chunklen = data["chunklen"]
    cparams = ca.cparams(
      clevel = data["cparams"]["clevel"],
      shuffle = data["cparams"]["shuffle"])
    expectedlen = data["expectedlen"]
    dflt = data["dflt"]
    return (shape, cparams, dtype_, dflt, expectedlen, cbytes, chunklen)

  def store_obj(self, object arrobj):
    cdef chunk chunk_
    import pickle

    pick_obj = pickle.dumps(arrobj, pickle.HIGHEST_PROTOCOL)
    chunk_ = chunk(pick_obj, np.dtype('O'), self._cparams,
                   _memory = self._rootdir is None)

    self.chunks.append(chunk_)
    # Update some counters
    nbytes, cbytes = chunk_.nbytes, chunk_.cbytes
    self._cbytes += cbytes
    self._nbytes += nbytes

  def append(self, object array):
    """
    append(array)

    Append a numpy `array` to this instance.

    Parameters
    ----------
    array : NumPy-like object
        The array to be appended.  Must be compatible with shape and type of
        the carray.

    """
    cdef int atomsize, itemsize, chunksize, leftover
    cdef int nbytesfirst, chunklen, start, stop
    cdef npy_intp nbytes, cbytes, bsize, i, nchunks
    cdef ndarray remainder, arrcpy, dflts
    cdef chunk chunk_

    if self.mode == "r":
      raise RuntimeError(
        "cannot modify data because mode is '%s'" % self.mode)

    arrcpy = utils.to_ndarray(array, self._dtype)
    if arrcpy.dtype != self._dtype.base:
      raise TypeError, "array dtype does not match with self"

    # Object dtype requires special storage
    if arrcpy.dtype.char == 'O':
      self.store_obj(array)
      return

    # Appending a single row should be supported
    if arrcpy.shape == self._dtype.shape:
      arrcpy = arrcpy.reshape((1,)+arrcpy.shape)
    if arrcpy.shape[1:] != self._dtype.shape:
      raise ValueError, "array trailing dimensions do not match with self"

    atomsize = self.atomsize
    itemsize = self.itemsize
    chunksize = self._chunksize
    chunks = self.chunks
    leftover = self.leftover
    bsize = arrcpy.size*itemsize
    cbytes = 0

    # Check if array fits in existing buffer
    if (bsize + leftover) < chunksize:
      # Data fits in lastchunk buffer.  Just copy it
      if arrcpy.strides[0] > 0:
        memcpy(self.lastchunk+leftover, arrcpy.data, bsize)
      else:
        start = cython.cdiv(leftover, atomsize)
        stop = cython.cdiv((leftover+bsize), atomsize)
        self.lastchunkarr[start:stop] = arrcpy
      leftover += bsize
    else:
      # Data does not fit in buffer.  Break it in chunks.

      # First, fill the last buffer completely (if needed)
      if leftover:
        nbytesfirst = chunksize - leftover
        if arrcpy.strides[0] > 0:
          memcpy(self.lastchunk+leftover, arrcpy.data, nbytesfirst)
        else:
          start = cython.cdiv(leftover, atomsize)
          stop = cython.cdiv((leftover+nbytesfirst), atomsize)
          self.lastchunkarr[start:stop] = arrcpy[start:stop]
        # Compress the last chunk and add it to the list
        chunk_ = chunk(self.lastchunkarr, self._dtype, self._cparams,
                       _memory = self._rootdir is None)
        chunks.append(chunk_)
        cbytes = chunk_.cbytes
      else:
        nbytesfirst = 0

      # Then fill other possible chunks
      nbytes = bsize - nbytesfirst
      nchunks = cython.cdiv(nbytes, <npy_intp>chunksize)
      chunklen = self._chunklen
      # Get a new view skipping the elements that have been already copied
      remainder = arrcpy[cython.cdiv(nbytesfirst, atomsize):]
      for i from 0 <= i < nchunks:
        chunk_ = chunk(
          remainder[i*chunklen:(i+1)*chunklen], self._dtype, self._cparams,
          _memory = self._rootdir is None)
        chunks.append(chunk_)
        cbytes += chunk_.cbytes

      # Finally, deal with the leftover
      leftover = nbytes % chunksize
      if leftover:
        remainder = remainder[nchunks*chunklen:]
        if arrcpy.strides[0] > 0:
          memcpy(self.lastchunk, remainder.data, leftover)
        else:
          self.lastchunkarr[:len(remainder)] = remainder

    # Update some counters
    self.leftover = leftover
    self._cbytes += cbytes
    self._nbytes += bsize
    return

  def trim(self, object nitems):
    """
    trim(nitems)

    Remove the trailing `nitems` from this instance.

    Parameters
    ----------
    nitems : int
        The number of trailing items to be trimmed.  If negative, the object
        is enlarged instead.

    """
    cdef int atomsize, leftover, leftover2
    cdef npy_intp cbytes, bsize, nchunk2
    cdef chunk chunk_

    if not isinstance(nitems, (int, long, float)):
      raise TypeError, "`nitems` must be an integer"

    # Check that we don't run out of space
    if nitems > self.len:
      raise ValueError, "`nitems` must be less than total length"
    # A negative number of items means that we want to grow the object
    if nitems <= 0:
      self.resize(self.len - nitems)
      return

    atomsize = self.atomsize
    chunks = self.chunks
    leftover = self.leftover
    bsize = nitems * atomsize
    cbytes = 0

    # Check if items belong to the last chunk
    if (leftover - bsize) > 0:
      # Just update leftover counter
      leftover -= bsize
    else:
      # nitems larger than last chunk
      nchunk = cython.cdiv((self.len - nitems), self._chunklen)
      leftover2 = (self.len - nitems) % self._chunklen
      leftover = leftover2 * atomsize

      # Remove complete chunks
      nchunk2 = lnchunk = cython.cdiv(self._nbytes, <npy_intp>self._chunksize)
      while nchunk2 > nchunk:
        chunk_ = chunks.pop()
        cbytes += chunk_.cbytes
        nchunk2 -= 1

      # Finally, deal with the leftover
      if leftover:
        self.lastchunkarr[:leftover2] = chunk_[:leftover2]
        if self._rootdir:
          # Last chunk is removed automatically by the chunks.pop() call, and
          # always is counted as if it is not compressed (although it is in
          # this state on-disk)
          cbytes += chunk_.nbytes

    # Update some counters
    self.leftover = leftover
    self._cbytes -= cbytes
    self._nbytes -= bsize
    # Flush last chunk and update counters on-disk
    self.flush()

  def resize(self, object nitems):
    """
    resize(nitems)

    Resize the instance to have `nitems`.

    Parameters
    ----------
    nitems : int
        The final length of the object.  If `nitems` is larger than the actual
        length, new items will appended using `self.dflt` as filling values.

    """
    cdef object chunk

    if not isinstance(nitems, (int, long, float)):
      raise TypeError, "`nitems` must be an integer"

    if nitems == self.len:
      return
    elif nitems < 0:
      raise ValueError, "`nitems` cannot be negative"

    if nitems > self.len:
      # Create a 0-strided array and append it to self
      chunk = np.ndarray(nitems-self.len, dtype=self._dtype,
                         buffer=self._dflt, strides=(0,))
      self.append(chunk)
      self.flush()
    else:
      # Just trim the excess of items
      self.trim(self.len-nitems)

  def reshape(self, newshape):
    """
    reshape(newshape)

    Returns a new carray containing the same data with a new shape.

    Parameters
    ----------
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is inferred
        from the length of the array and remaining dimensions.

    Returns
    -------
    reshaped_array : carray
        A copy of the original carray.

    """
    cdef npy_intp newlen, ilen, isize, osize, newsize, rsize, i
    cdef object ishape, oshape, pos, newdtype, out

    # Enforce newshape as tuple
    if isinstance(newshape, (int, long)):
      newshape = (newshape,)
    newsize = np.prod(newshape)

    ishape = self.shape
    ilen = ishape[0]
    isize = np.prod(ishape)

    # Check for -1 in newshape
    if -1 in newshape:
      if newshape.count(-1) > 1:
        raise ValueError, "only one shape dimension can be -1"
      pos = newshape.index(-1)
      osize = np.prod(newshape[:pos] + newshape[pos+1:])
      if isize == 0:
        newshape = newshape[:pos] + (0,) + newshape[pos+1:]
      else:
        newshape = newshape[:pos] + (isize/osize,) + newshape[pos+1:]
      newsize = np.prod(newshape)

    # Check shape compatibility
    if isize != newsize:
      raise ValueError, "`newshape` is not compatible with the current one"
    # Create the output container
    newdtype = np.dtype((self._dtype.base, newshape[1:]))
    newlen = newshape[0]

    # If shapes are both n-dimensional, convert first to 1-dim shape
    # and then convert again to the final newshape.
    if len(ishape) > 1 and len(newshape) > 1:
      out = self.reshape(-1)
      return out.reshape(newshape)

    if self._rootdir:
      # If persistent, do the copy to a temporary dir
      absdir = os.path.dirname(self._rootdir)
      rootdir = tempfile.mkdtemp(suffix='__temp__', dir=absdir)
    else:
      rootdir = None

    # Create the final container and fill it
    out = carray([], dtype=newdtype, cparams=self.cparams, expectedlen=newlen,
                 rootdir=rootdir, mode='w')
    if newlen < ilen:
      rsize = isize / newlen
      for i from 0 <= i < newlen:
        out.append(self[i*rsize:(i+1)*rsize].reshape(newdtype.shape))
    else:
      for i from 0 <= i < ilen:
        out.append(self[i].reshape(-1))
    out.flush()

    # Finally, rename the temporary data directory to self._rootdir
    if self._rootdir:
      shutil.rmtree(self._rootdir)
      os.rename(rootdir, self._rootdir)
      # Restore the rootdir and mode
      out.rootdir = self._rootdir
      out.mode = self._mode

    return out

  def copy(self, **kwargs):
    """
    copy(**kwargs)

    Return a copy of this object.

    Parameters
    ----------
    kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns
    -------
    out : carray object
        The copy of this object.

    """
    cdef object chunklen

    # Get defaults for some parameters
    cparams = kwargs.pop('cparams', self._cparams)
    expectedlen = kwargs.pop('expectedlen', self.len)

    # Create a new, empty carray
    ccopy = carray(np.empty(0, dtype=self._dtype),
                   cparams=cparams,
                   expectedlen=expectedlen,
                   **kwargs)

    # Now copy the carray chunk by chunk
    chunklen = self._chunklen
    for i from 0 <= i < self.len by chunklen:
      ccopy.append(self[i:i+chunklen])
    ccopy.flush()

    return ccopy

  def sum(self, dtype=None):
    """
    sum(dtype=None)

    Return the sum of the array elements.

    Parameters
    ----------
    dtype : NumPy dtype
        The desired type of the output.  If ``None``, the dtype of `self` is
        used.  An exception is when `self` has an integer type with less
        precision than the default platform integer.  In that case, the
        default platform integer is used instead (NumPy convention).


    Return value
    ------------
    out : NumPy scalar with `dtype`

    """
    cdef chunk chunk_
    cdef npy_intp nchunk, nchunks
    cdef object result

    if dtype is None:
      dtype = self._dtype.base
      # Check if we have less precision than required for ints
      # (mimick NumPy logic)
      if dtype.kind in ('b', 'i') and dtype.itemsize < IntType.itemsize:
        dtype = IntType
    else:
      dtype = np.dtype(dtype)
    if dtype.kind == 'S':
      raise TypeError, "cannot perform reduce with flexible type"

    # Get a container for the result
    result = np.zeros(1, dtype=dtype)[0]

    nchunks = cython.cdiv(self._nbytes, <npy_intp>self._chunksize)
    for nchunk from 0 <= nchunk < nchunks:
      chunk_ = self.chunks[nchunk]
      if chunk_.isconstant:
        result += chunk_.constant * self._chunklen
      elif self._dtype.type == np.bool_:
        result += chunk_.true_count
      else:
        result += chunk_[:].sum(dtype=dtype)
    if self.leftover:
      leftover = self.len - nchunks * self._chunklen
      result += self.lastchunkarr[:leftover].sum(dtype=dtype)

    return result

  def __len__(self):
    return self.len

  def __sizeof__(self):
    return self._cbytes

  cdef int getitem_cache(self, npy_intp pos, char *dest):
    """Get a single item and put it in `dest`.  It caches a complete block.

    It returns 1 if asked `pos` can be copied to `dest`.  Else, this returns
    0.

    NOTE: As Blosc supports decompressing just a block inside a chunk, the
    data that is cached is a *block*, as it is the least amount of data that
    can be decompressed.  This saves both time and memory.

    IMPORTANT: Any update operation (e.g. __setitem__) *must* disable this
    cache by setting self.idxcache = -2.
    """
    cdef int ret, atomsize, blocksize, offset
    cdef int idxcache, posinbytes, blocklen
    cdef npy_intp nchunk, nchunks, chunklen
    cdef chunk chunk_

    atomsize = self.atomsize
    nchunks = cython.cdiv(self._nbytes, <npy_intp>self._chunksize)
    chunklen = self._chunklen
    nchunk = cython.cdiv(pos, <npy_intp>chunklen)

    # Check whether pos is in the last chunk
    if nchunk == nchunks and self.leftover:
      posinbytes = (pos % chunklen) * atomsize
      memcpy(dest, self.lastchunk + posinbytes, atomsize)
      return 1

    # Locate the *block* inside the chunk
    chunk_ = self.chunks[nchunk]
    blocksize = chunk_.blocksize
    blocklen = cython.cdiv(blocksize, atomsize)

    if atomsize > blocksize:
      # This request cannot be resolved here
      return 0

    # Check whether the cache block has to be initialized
    if self.idxcache < 0:
      self.blockcache = np.empty(shape=(blocklen,), dtype=self._dtype)
      self.datacache = self.blockcache.data
      # We don't want this to contribute to cbytes counter!
      # if self.idxcache == -1:
      #   # Absolute first time.  Add the cache size to cbytes counter.
      #   self._cbytes += chunksize

    # Check if block is cached
    idxcache = cython.cdiv(pos, <npy_intp>blocklen) * blocklen
    if idxcache == self.idxcache:
      # Hit!
      posinbytes = (pos % blocklen) * atomsize
      memcpy(dest, self.datacache + posinbytes, atomsize)
      return 1

    # No luck. Read a complete block.
    offset = idxcache % chunklen
    chunk_._getitem(offset, offset+blocklen, self.datacache)
    # Copy the interesting bits to dest
    posinbytes = (pos % blocklen) * atomsize
    memcpy(dest, self.datacache + posinbytes, atomsize)
    # Update the cache index
    self.idxcache = idxcache
    return 1

  def getitem_object(self, start, stop=None, step=None):
    """Retrieve elements of type object."""
    import pickle

    if stop is None and step is None:
      # Integer
      cchunk = self.chunks[start]
      chunk = cchunk.getudata()
      return pickle.loads(chunk)

    # Range
    objs = [self.getitem_object(i) for i in xrange(start, stop, step)]
    return np.array(objs, dtype=self._dtype)

  def __getitem__(self, object key):
    """
    x.__getitem__(key) <==> x[key]

    Returns values based on `key`.  All the functionality of
    ``ndarray.__getitem__()`` is supported (including fancy indexing), plus a
    special support for expressions:

    Parameters
    ----------
    key : string
        It will be interpret as a boolean expression (computed via `eval`) and
        the elements where these values are true will be returned as a NumPy
        array.

    See Also
    --------
    eval

    """

    cdef int chunklen
    cdef npy_intp startb, stopb
    cdef npy_intp nchunk, keychunk, nchunks
    cdef npy_intp nwrow, blen
    cdef ndarray arr1
    cdef object start, stop, step
    cdef object arr

    chunklen = self._chunklen

    # Check for integer
    # isinstance(key, int) is not enough in Cython (?)
    if isinstance(key, (int, long)) or isinstance(key, np.int_):
      if key < 0:
        # To support negative values
        key += self.len
      if key >= self.len:
        raise IndexError, "index out of range"
      arr1 = self.arr1
      if self.dtype.char == 'O':
        return self.getitem_object(key)
      if self.getitem_cache(key, arr1.data):
        if self.itemsize == self.atomsize:
          return PyArray_GETITEM(arr1, arr1.data)
        else:
          return arr1[0]
      # Fallback action: use the slice code
      return np.squeeze(self[slice(key, None, 1)])
    # Slices
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
      if step and step <= 0 :
        raise NotImplementedError("step in slice can only be positive")
    # Multidimensional keys
    elif isinstance(key, tuple):
      if len(key) == 0:
        raise ValueError("empty tuple not supported")
      elif len(key) == 1:
        return self[key[0]]
      # An n-dimensional slice
      # First, retrieve elements in the leading dimension
      arr = self[key[0]]
      # Then, keep only the required elements in other dimensions
      if type(key[0]) == slice:
        arr = arr[(slice(None),) + key[1:]]
      else:
        arr = arr[key[1:]]
      # Force a copy in case returned array is not contiguous
      if not arr.flags.contiguous:
        arr = arr.copy()
      return arr
    # List of integers (case of fancy indexing)
    elif isinstance(key, list):
      # Try to convert to a integer array
      try:
        key = np.array(key, dtype=np.int_)
      except:
        raise IndexError, "key cannot be converted to an array of indices"
      return self[key]
    # A boolean or integer array (case of fancy indexing)
    elif hasattr(key, "dtype"):
      if key.dtype.type == np.bool_:
        # A boolean array
        if len(key) != self.len:
          raise IndexError, "boolean array length must match len(self)"
        if isinstance(key, carray):
          count = key.sum()
        else:
          count = -1
        return np.fromiter(self.where(key), dtype=self._dtype, count=count)
      elif np.issubsctype(key, np.int_):
        # An integer array
        return np.array([self[i] for i in key], dtype=self._dtype)
      else:
        raise IndexError, \
              "arrays used as indices must be of integer (or boolean) type"
    # An boolean expression (case of fancy indexing)
    elif type(key) is str:
      # Evaluate
      result = ca.eval(key)
      if result.dtype.type != np.bool_:
        raise IndexError, "only boolean expressions supported"
      if len(result) != self.len:
        raise IndexError, "boolean expression outcome must match len(self)"
      # Call __getitem__ again
      return self[result]
    # All the rest not implemented
    else:
      raise NotImplementedError, "key not supported: %s" % repr(key)

    # From now on, will only deal with [start:stop:step] slices

    # Get the corrected values for start, stop, step
    (start, stop, step) = slice(start, stop, step).indices(self.len)

    # Build a numpy container
    blen = get_len_of_range(start, stop, step)
    arr = np.empty(shape=(blen,), dtype=self._dtype)
    if blen == 0:
      # If empty, return immediately
      return arr

    if self.dtype.char == 'O':
      return self.getitem_object(start, stop, step)

    # Fill it from data in chunks
    nwrow = 0
    nchunks = cython.cdiv(self._nbytes, <npy_intp>self._chunksize)
    if self.leftover > 0:
      nchunks += 1
    for nchunk from 0 <= nchunk < nchunks:
      # Compute start & stop for each block
      startb, stopb, blen = clip_chunk(nchunk, chunklen, start, stop, step)
      if blen == 0:
        continue
      # Get the data chunk and assign it to result array
      if nchunk == nchunks-1 and self.leftover:
        arr[nwrow:nwrow+blen] = self.lastchunkarr[startb:stopb:step]
      else:
        arr[nwrow:nwrow+blen] = self.chunks[nchunk][startb:stopb:step]
      nwrow += blen

    return arr

  def __setitem__(self, object key, object value):
    """
    x.__setitem__(key, value) <==> x[key] = value

    Sets values based on `key`.  All the functionality of
    ``ndarray.__setitem__()`` is supported (including fancy indexing), plus a
    special support for expressions:

    Parameters
    ----------
    key : string
        It will be interpret as a boolean expression (computed via `eval`) and
        the elements where these values are true will be set to `value`.

    See Also
    --------
    eval

    """
    cdef int chunklen
    cdef npy_intp startb, stopb
    cdef npy_intp nchunk, keychunk, nchunks
    cdef npy_intp nwrow, blen, vlen
    cdef chunk chunk_
    cdef object start, stop, step
    cdef object cdata, arr

    if self.mode == "r":
      raise RuntimeError(
        "cannot modify data because mode is '%s'" % self.mode)

    # We are going to modify data.  Mark block cache as dirty.
    if self.idxcache >= 0:
      # -2 means that cbytes counter has not to be changed
      self.idxcache = -2

    # Check for integer
    # isinstance(key, int) is not enough in Cython (?)
    if isinstance(key, (int, long)) or isinstance(key, np.int_):
      if key < 0:
        # To support negative values
        key += self.len
      if key >= self.len:
        raise IndexError, "index out of range"
      (start, stop, step) = key, key+1, 1
    # Slices
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
      if step:
        if step <= 0 :
          raise NotImplementedError("step in slice can only be positive")
    # Multidimensional keys
    elif isinstance(key, tuple):
      if len(key) == 0:
        raise ValueError("empty tuple not supported")
      elif len(key) == 1:
        self[key[0]] = value
        return
      # An n-dimensional slice
      # First, retrieve elements in the leading dimension
      arr = self[key[0]]
      # Then, assing only the requested elements in other dimensions
      if type(key[0]) == slice:
        arr[(slice(None),) + key[1:]] = value
      else:
        arr[key[1:]] = value
      # Finally, update this superset of values in self
      self[key[0]] = arr
      return
    # List of integers (case of fancy indexing)
    elif isinstance(key, list):
      # Try to convert to a integer array
      try:
        key = np.array(key, dtype=np.int_)
      except:
        raise IndexError, "key cannot be converted to an array of indices"
      self[key] = value
      return
    # A boolean or integer array (case of fancy indexing)
    elif hasattr(key, "dtype"):
      if key.dtype.type == np.bool_:
        # A boolean array
        if len(key) != self.len:
          raise ValueError, "boolean array length must match len(self)"
        self.bool_update(key, value)
        return
      elif np.issubsctype(key, np.int_):
        # An integer array
        value = utils.to_ndarray(value, self._dtype, arrlen=len(key))
        # XXX This could be optimised, but it works like this
        for i, item in enumerate(key):
          self[item] = value[i]
        return
      else:
        raise IndexError, \
              "arrays used as indices must be of integer (or boolean) type"
    # An boolean expression (case of fancy indexing)
    elif type(key) is str:
      # Evaluate
      result = ca.eval(key)
      if result.dtype.type != np.bool_:
        raise IndexError, "only boolean expressions supported"
      if len(result) != self.len:
        raise IndexError, "boolean expression outcome must match len(self)"
      # Call __setitem__ again
      self[result] = value
      return
    # All the rest not implemented
    else:
      raise NotImplementedError, "key not supported: %s" % repr(key)

    # Get the corrected values for start, stop, step
    (start, stop, step) = slice(start, stop, step).indices(self.len)

    # Build a numpy object out of value
    vlen = get_len_of_range(start, stop, step)
    if vlen == 0:
      # If range is empty, return immediately
      return
    value = utils.to_ndarray(value, self._dtype, arrlen=vlen)

    # Fill it from data in chunks
    nwrow = 0
    chunklen = self._chunklen
    nchunks = cython.cdiv(self._nbytes, <npy_intp>self._chunksize)
    if self.leftover > 0:
      nchunks += 1
    for nchunk from 0 <= nchunk < nchunks:
      # Compute start & stop for each block
      startb, stopb, blen = clip_chunk(nchunk, chunklen, start, stop, step)
      if blen == 0:
        continue
      # Modify the data in chunk
      if nchunk == nchunks-1 and self.leftover:
        self.lastchunkarr[startb:stopb:step] = value[nwrow:nwrow+blen]
      else:
        # Get the data chunk
        chunk_ = self.chunks[nchunk]
        self._cbytes -= chunk_.cbytes
        # Get all the values there
        cdata = chunk_[:]
        # Overwrite it with data from value
        cdata[startb:stopb:step] = value[nwrow:nwrow+blen]
        # Replace the chunk
        chunk_ = chunk(cdata, self._dtype, self._cparams,
                       _memory = self._rootdir is None)
        self.chunks[nchunk] = chunk_
        # Update cbytes counter
        self._cbytes += chunk_.cbytes
      nwrow += blen

    # Safety check
    assert (nwrow == vlen)

  # This is a private function that is specific for `eval`
  def _getrange(self, npy_intp start, npy_intp blen, ndarray out):
    cdef int chunklen
    cdef npy_intp startb, stopb
    cdef npy_intp nwrow, stop, cblen
    cdef npy_intp schunk, echunk, nchunk, nchunks
    cdef chunk chunk_

    # Check that we are inside limits
    nrows = cython.cdiv(self._nbytes, <npy_intp>self.atomsize)
    if (start + blen) > nrows:
      blen = nrows - start

    # Fill `out` from data in chunks
    nwrow = 0
    stop = start + blen
    nchunks = cython.cdiv(self._nbytes, <npy_intp>self._chunksize)
    chunklen = cython.cdiv(self._chunksize, self.atomsize)
    schunk = cython.cdiv(start, <npy_intp>chunklen)
    echunk = cython.cdiv((start+blen), <npy_intp>chunklen)
    for nchunk from schunk <= nchunk <= echunk:
      # Compute start & stop for each block
      startb = start % chunklen
      stopb = chunklen
      if (start + startb) + chunklen > stop:
        # XXX I still have to explain why this expression works
        # for chunklen > (start + blen)
        stopb = (stop - start) + startb
      cblen = stopb - startb
      if cblen == 0:
        continue
      # Get the data chunk and assign it to result array
      if nchunk == nchunks and self.leftover:
        out[nwrow:nwrow+cblen] = self.lastchunkarr[startb:stopb]
      else:
        chunk_ = self.chunks[nchunk]
        chunk_._getitem(startb, stopb, out.data+nwrow*self.atomsize)
      nwrow += cblen
      start += cblen

  cdef void bool_update(self, boolarr, value):
    """Update self in positions where `boolarr` is true with `value` array."""
    cdef int chunklen
    cdef npy_intp startb, stopb
    cdef npy_intp nchunk, nchunks, nrows
    cdef npy_intp nwrow, blen, vlen, n
    cdef chunk chunk_
    cdef object cdata, boolb

    vlen = boolarr.sum()   # number of true values in bool array
    value = utils.to_ndarray(value, self._dtype, arrlen=vlen)

    # Fill it from data in chunks
    nwrow = 0
    chunklen = self._chunklen
    nchunks = cython.cdiv(self._nbytes, <npy_intp>self._chunksize)
    if self.leftover > 0:
      nchunks += 1
    nrows = cython.cdiv(self._nbytes, <npy_intp>self.atomsize)
    for nchunk from 0 <= nchunk < nchunks:
      # Compute start & stop for each block
      startb, stopb, _ = clip_chunk(nchunk, chunklen, 0, nrows, 1)
      # Get boolean values for this chunk
      n = nchunk * chunklen
      boolb = boolarr[n+startb:n+stopb]
      blen = boolb.sum()
      if blen == 0:
        continue
      # Modify the data in chunk
      if nchunk == nchunks-1 and self.leftover:
        self.lastchunkarr[boolb] = value[nwrow:nwrow+blen]
      else:
        # Get the data chunk
        chunk_ = self.chunks[nchunk]
        self._cbytes -= chunk_.cbytes
        # Get all the values there
        cdata = chunk_[:]
        # Overwrite it with data from value
        cdata[boolb] = value[nwrow:nwrow+blen]
        # Replace the chunk
        chunk_ = chunk(cdata, self._dtype, self._cparams,
                       _memory = self._rootdir is None)
        self.chunks[nchunk] = chunk_
        # Update cbytes counter
        self._cbytes += chunk_.cbytes
      nwrow += blen

    # Safety check
    assert (nwrow == vlen)

  def __iter__(self):

    if not self.sss_mode:
      self.start = 0
      self.stop = cython.cdiv(self._nbytes, <npy_intp>self.atomsize)
      self.step = 1
    if not (self.sss_mode or self.where_mode or self.wheretrue_mode):
      self.nhits = 0
      self.limit = sys.maxint
      self.skip = 0
    # Initialize some internal values
    self.startb = 0
    self.nrowsread = self.start
    self._nrow = self.start - self.step
    self._row = -1  # a sentinel
    if self.where_mode and isinstance(self.where_arr, carray):
      self.nrowsinbuf = self.where_arr.chunklen
    else:
      self.nrowsinbuf = self._chunklen

    return self

  def iter(self, start=0, stop=None, step=1, limit=None, skip=0):
    """
    iter(start=0, stop=None, step=1, limit=None, skip=0)

    Iterator with `start`, `stop` and `step` bounds.

    Parameters
    ----------
    start : int
        The starting item.
    stop : int
        The item after which the iterator stops.
    step : int
        The number of items incremented during each iteration.  Cannot be
        negative.
    limit : int
        A maximum number of elements to return.  The default is return
        everything.
    skip : int
        An initial number of elements to skip.  The default is 0.

    Returns
    -------
    out : iterator

    See Also
    --------
    where, wheretrue

    """
    # Check limits
    if step <= 0:
      raise NotImplementedError, "step param can only be positive"
    self.start, self.stop, self.step = \
        slice(start, stop, step).indices(self.len)
    self.reset_sentinels()
    self.sss_mode = True
    if limit is not None:
      self.limit = limit + skip
    self.skip = skip
    return iter(self)

  def wheretrue(self, limit=None, skip=0):
    """
    wheretrue(limit=None, skip=0)

    Iterator that returns indices where this object is true.

    This is currently only useful for boolean carrays that are unidimensional.

    Parameters
    ----------
    limit : int
        A maximum number of elements to return.  The default is return
        everything.
    skip : int
        An initial number of elements to skip.  The default is 0.

    Returns
    -------
    out : iterator

    See Also
    --------
    iter, where

    """
    # Check self
    if self._dtype.base.type != np.bool_:
      raise ValueError, "`self` is not an array of booleans"
    if self.ndim > 1:
      raise NotImplementedError, "`self` is not unidimensional"
    self.reset_sentinels()
    self.wheretrue_mode = True
    if limit is not None:
      self.limit = limit + skip
    self.skip = skip
    return iter(self)

  def where(self, boolarr, limit=None, skip=0):
    """
    where(boolarr, limit=None, skip=0)

    Iterator that returns values of this object where `boolarr` is true.

    This is currently only useful for boolean carrays that are unidimensional.

    Parameters
    ----------
    boolarr : a carray or NumPy array of boolean type
        The boolean values.
    limit : int
        A maximum number of elements to return.  The default is return
        everything.
    skip : int
        An initial number of elements to skip.  The default is 0.

    Returns
    -------
    out : iterator

    See Also
    --------
    iter, wheretrue

    """
    # Check input
    if self.ndim > 1:
      raise NotImplementedError, "`self` is not unidimensional"
    if not hasattr(boolarr, "dtype"):
      raise ValueError, "`boolarr` is not an array"
    if boolarr.dtype.type != np.bool_:
      raise ValueError, "`boolarr` is not an array of booleans"
    if len(boolarr) != self.len:
      raise ValueError, "`boolarr` must be of the same length than ``self``"
    self.reset_sentinels()
    self.where_mode = True
    self.where_arr = boolarr
    if limit is not None:
      self.limit = limit + skip
    self.skip = skip
    return iter(self)

  def __next__(self):
    cdef char *vbool
    cdef int nhits_buf

    self.nextelement = self._nrow + self.step
    while (self.nextelement < self.stop) and (self.nhits < self.limit):
      if self.nextelement >= self.nrowsread:
        # Skip until there is interesting information
        while self.nextelement >= self.nrowsread + self.nrowsinbuf:
          self.nrowsread += self.nrowsinbuf
        # Compute the end for this iteration
        self.stopb = self.stop - self.nrowsread
        if self.stopb > self.nrowsinbuf:
          self.stopb = self.nrowsinbuf
        self._row = self.startb - self.step

        # Skip chunks with zeros if in wheretrue_mode
        if self.wheretrue_mode and self.check_zeros(self):
          self.nrowsread += self.nrowsinbuf
          self.nextelement += self.nrowsinbuf
          continue

        if self.where_mode:
          # Skip chunks with zeros in where_arr
          if self.check_zeros(self.where_arr):
            self.nrowsread += self.nrowsinbuf
            self.nextelement += self.nrowsinbuf
            continue
          # Read a chunk of the boolean array
          self.where_buf = self.where_arr[
            self.nrowsread:self.nrowsread+self.nrowsinbuf]

        # Read a data chunk
        self.iobuf = self[self.nrowsread:self.nrowsread+self.nrowsinbuf]
        self.nrowsread += self.nrowsinbuf

        # Check if we can skip this buffer
        if (self.wheretrue_mode or self.where_mode) and self.skip > 0:
          if self.wheretrue_mode:
            nhits_buf = self.iobuf.sum()
          else:
            nhits_buf = self.where_buf.sum()
          if (self.nhits + nhits_buf) < self.skip:
            self.nhits += nhits_buf
            self.nextelement += self.nrowsinbuf
            continue

      self._row += self.step
      self._nrow = self.nextelement
      if self._row + self.step >= self.stopb:
        # Compute the start row for the next buffer
        self.startb = (self._row + self.step) % self.nrowsinbuf
      self.nextelement = self._nrow + self.step

      # Return a value depending on the mode we are
      if self.wheretrue_mode:
        vbool = <char *>(self.iobuf.data + self._row)
        if vbool[0]:
          self.nhits += 1
          if self.nhits <= self.skip:
            continue
          return self._nrow
        else:
          continue
      if self.where_mode:
        vbool = <char *>(self.where_buf.data + self._row)
        if not vbool[0]:
            continue
      self.nhits += 1
      if self.nhits <= self.skip:
        continue
      # Return the current value in I/O buffer
      if self.itemsize == self.atomsize:
        return PyArray_GETITEM(
          self.iobuf, self.iobuf.data + self._row * self.atomsize)
      else:
        return self.iobuf[self._row]

    else:
      # Release buffers
      self.iobuf = np.empty(0, dtype=self._dtype)
      self.where_buf = np.empty(0, dtype=np.bool_)
      self.reset_sentinels()
      raise StopIteration        # end of iteration

  cdef reset_sentinels(self):
    """Reset sentinels for iterator."""
    self.sss_mode = False
    self.wheretrue_mode = False
    self.where_mode = False
    self.where_arr = None
    self.nhits = 0
    self.limit = sys.maxint
    self.skip = 0

  cdef int check_zeros(self, object barr):
    """Check for zeros.  Return 1 if all zeros, else return 0."""
    cdef int bsize
    cdef npy_intp nchunk
    cdef carray carr
    cdef ndarray ndarr
    cdef chunk chunk_

    if isinstance(barr, carray):
      # Check for zero'ed chunks in carrays
      carr = barr
      nchunk = cython.cdiv(self.nrowsread, <npy_intp>self.nrowsinbuf)
      if nchunk < len(carr.chunks):
        chunk_ = carr.chunks[nchunk]
        if chunk_.isconstant and chunk_.constant in (0, ''):
          return 1
    else:
      # Check for zero'ed chunks in ndarrays
      ndarr = barr
      bsize = self.nrowsinbuf
      if self.nrowsread + bsize > self.len:
        bsize = self.len - self.nrowsread
      if check_zeros(ndarr.data + self.nrowsread, bsize):
        return 1
    return 0

  def _update_disk_sizes(self):
    """Update the sizes on-disk."""
    sizes = dict()
    if self._rootdir:
      sizes['shape'] = self.shape
      sizes['nbytes'] = self.nbytes
      sizes['cbytes'] = self.cbytes
      rowsf = os.path.join(self.metadir, SIZES_FILE)
      with open(rowsf, 'wb') as rowsfh:
        rowsfh.write(json.dumps(sizes))
        rowsfh.write('\n')

  def flush(self):
    """Flush data in internal buffers to disk.

    This call should typically be done after performing modifications
    (__settitem__(), append()) in persistence mode.  If you don't do this, you
    risk loosing part of your modifications.

    """
    cdef chunk chunk_
    cdef npy_intp nchunks
    cdef int leftover_atoms

    if self._rootdir is None:
      return

    if self.leftover:
      leftover_atoms = cython.cdiv(self.leftover, self.atomsize)
      chunk_ = chunk(self.lastchunkarr[:leftover_atoms], self.dtype,
                     self.cparams,
                     _memory = self._rootdir is None)
      # Flush this chunk to disk
      self.chunks.flush(chunk_)

    # Finally, update the sizes metadata on-disk
    self._update_disk_sizes()

  # XXX This does not work.  Will have to realize how to properly
  # flush buffers before self going away...
  # def __del__(self):
  #   # Make a flush to disk if this object get disposed
  #   self.flush()

  def __str__(self):
    return array2string(self)

  def __repr__(self):
    snbytes = utils.human_readable_size(self._nbytes)
    scbytes = utils.human_readable_size(self._cbytes)
    cratio = self._nbytes / float(self._cbytes)
    header = "carray(%s, %s)\n" % (self.shape, self.dtype)
    header += "  nbytes: %s; cbytes: %s; ratio: %.2f\n" % (
      snbytes, scbytes, cratio)
    header += "  cparams := %r\n" % self.cparams
    if self._rootdir:
      header += "  rootdir := '%s'\n" % self._rootdir
    fullrepr = header + str(self)
    return fullrepr



## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
