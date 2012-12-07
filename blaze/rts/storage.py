"""
Blaze storage backend. We follow the same conventions as CPython in that
we use the arena model of memory.

It does the mallocs of data as needed to shuffle blocks efficiently for
data that is passed to Numba. It is also one place for safe management
of blocks of data from SQL, Disk, IOPro, etc to allocate on to rather
than having every adaptor define memory wherever it feels like calling
malloc.

:Inspiration:

    http://svn.python.org/projects/python/trunk/Python/pyarena.c

:Design Principles:

- Will write a Cython wrapper that should be drop in
  replacable for ``libc.stdlib.malloc``.

- Be able to do zero copy network transfers straight into arena
  buffers from ZeroMQ and MPI.

- Be able store the data on the GPU while executing Numba GPU kernels.
  The problem is the same as a "distributed memory" problem over since we
  have two storage substrates Siu will probably have more insight on how
  this should be designed.

- Will almost certainly migrate this to C / Cython

"""

import os
import sys
import mmap
import ctypes
import bisect
import weakref
import traceback
import threading
import itertools
import numpy as np

from blaze.cutils import buffer_pointer

def address_of_buffer(buf):
    if isinstance(buf, memoryview):
        return id(buf), len(buf)
    elif isinstance(buf, mmap.mmap):
        return buffer_pointer(buf)


ALIGN_L2 = 2**17
ALIGN_L3 = 2**20
ALIGN_PAGE = mmap.PAGESIZE

#------------------------------------------------------------------------
# Arenas
#------------------------------------------------------------------------

class Arena(object):

    def __init__(self, size, name=None):
        # malloc but with \x00
        self.block = mmap.mmap(-1, size)

        self.size = size
        self.name = None

    def write_raw(self, by):
        assert isinstance(by, bytes)

#------------------------------------------------------------------------
# Heap
#------------------------------------------------------------------------

class Heap(object):

    _alignment = 8

    def __init__(self, size=mmap.PAGESIZE, align=ALIGN_PAGE):
        self.align = align

        self._lock = threading.Lock()
        self._size = size
        self._lengths = []
        self._len_to_seq = {}
        self._start_to_block = {}
        self._stop_to_block = {}
        self._allocated_blocks = set()
        self._arenas = []
        self._finalizers = {}

    @staticmethod
    def _roundup(n, alignment):
        mask = alignment - 1
        return (n + mask) & ~mask

    def free(self, block):
        # free a block returned by malloc()
        self._lock.acquire()
        try:
            self._allocated_blocks.remove(block)
            self._free(block)
        finally:
            self._lock.release()

    def malloc(self, size):
        # return a block of right size (possibly rounded up)
        assert 0 <= size < sys.maxint
        self._lock.acquire()
        try:
            size = self._roundup(max(size,1), self._alignment)
            (arena, start, stop) = self._malloc(size)
            new_stop = start + size
            if new_stop < stop:
                self._free((arena, new_stop, stop))
            block = (arena, start, new_stop)
            self._allocated_blocks.add(block)
            return block
        finally:
            self._lock.release()

    def _malloc(self, size):
        i = bisect.bisect_left(self._lengths, size)
        if i == len(self._lengths):
            length = self._roundup(max(self._size, size), self.align)
            self._size *= 2
            arena = Arena(length)
            self._arenas.append(arena)
            return (arena, 0, length)
        else:
            length = self._lengths[i]
            seq = self._len_to_seq[length]
            block = seq.pop()
            if not seq:
                del self._len_to_seq[length], self._lengths[i]

        (arena, start, stop) = block
        del self._start_to_block[(arena, start)]
        del self._stop_to_block[(arena, stop)]
        return block

    def _free(self, block):
        # free location and try to merge with neighbours
        (arena, start, stop) = block

        try:
            prev_block = self._stop_to_block[(arena, start)]
        except KeyError:
            pass
        else:
            start, _ = self._absorb(prev_block)

        try:
            next_block = self._start_to_block[(arena, stop)]
        except KeyError:
            pass
        else:
            _, stop = self._absorb(next_block)

        block = (arena, start, stop)
        length = stop - start

        try:
            self._len_to_seq[length].append(block)
        except KeyError:
            self._len_to_seq[length] = [block]
            bisect.insort(self._lengths, length)

        self._start_to_block[(arena, start)] = block
        self._stop_to_block[(arena, stop)] = block

    def _absorb(self, block):
        # deregister this block so it can be merged with a neighbour
        (arena, start, stop) = block
        del self._start_to_block[(arena, start)]
        del self._stop_to_block[(arena, stop)]

        length = stop - start
        seq = self._len_to_seq[length]
        seq.remove(block)
        if not seq:
            del self._len_to_seq[length]
            self._lengths.remove(length)

        return start, stop

#------------------------------------------------------------------------
# Heap Objects
#------------------------------------------------------------------------

class Buffer(object):

    def __init__(self, size, heap):
        assert 0 <= size < sys.maxint
        block = heap.malloc(size)
        self._state = (block, size)
        Finalizer(heap, self, heap.free, args=(block,))

    def get_address(self):
        (arena, start, stop), size = self._state
        address, length = buffer_pointer(arena.block)
        assert size <= length
        return address + start

    def get_size(self):
        return self._state[1]

def allocate_raw(heap, nbytes):
    buf = Buffer(nbytes, heap)
    address = buf.get_address()

    block, size = buf._state
    arena, start, new_stop = block

    return address, block, (ctypes.c_char*nbytes).from_address(address)

def allocate_numpy(heap, dtype, shape):
    """ Allocate a NumPy array conforming to the given shape on the heap """
    count = np.prod(shape)
    size = dtype.itemsize * count
    buf = Buffer(size, heap)
    address = buf.get_address()

    block, size = buf._state
    arena, start, new_stop = block

    return address, np.frombuffer(arena.block, dtype, count)

def allocate_carray(heap, dtype, chunksize):
    """ Allocate a buffer capable of holding a carray chunk """
    size = dtype.itemsize * chunksize
    buf = Buffer(size)
    address = buf.get_address()

    block, size = buf._state
    arena, start, new_stop = block

    return address, np.frombuffer(arena.block)

def numpy_pointer(numpy_array, ctype=ctypes.c_void_p):
    return numpy_array.ctypes.data_as(ctype)

#------------------------------------------------------------------------
# Finalizers
#------------------------------------------------------------------------

class Finalizer(object):
    def __init__(self, heap, obj, callback, args=(), kwargs=None):

        self._heap = heap
        self._callback = callback
        self._args = args
        self._kwargs = kwargs or {}
        self.ident = id(obj)
        heap._finalizers[self.ident] = self

    def __call__(self, wr=None):
        try:
            del self._heap._finalizers[self.ident]
        except KeyError:
            raise RuntimeError()
        else:
            self._callback(*self._args, **self._kwargs)
            self._weakref = None

def finalize(heap):
    items = [x for x in heap._finalizers.items()]
    error = None

    for key, finalizer in items:
        finalizer()

    if not error:
        for block in heap._allocated_blocks:
            heap.free(block)
    else:
        raise RuntimeError("Could not free blocks because finalizer failed")
