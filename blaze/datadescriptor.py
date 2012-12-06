def _dummy(*args, **kw):
    pass
Int = Float = Str = Tuple = Bool = List = _dummy

#------------------------------------------------------------------------
# Data Descriptor
#------------------------------------------------------------------------

class DataDescriptor(object):
    """ DataDescriptors are the underlying, low-level references to data
    that is returned by manifest Indexable objects (i.e. objects backed
    by real data of some variety).

    Whereas traditional data interfaces use iterators to programmatically
    retrieve data, Blaze preserves the ability to expose data in bulk
    form at as low a level as possible.
    """

    def __init__(self, id, nbytes):
        # XXX: whatever, just something unique for now
        self.id = id
        self.nbytes = nbytes

    def __str__(self):
        return self.id

    def __repr__(self):
        return self.id

    #------------------------------------------------------------------------
    # Generic adapters
    #------------------------------------------------------------------------

    def asbuffer(self, copy=False):
        """ Returns the buffer as a memoryview. If **copy** is False, then the
        memoryview just references the underlying data if possible.  If
        **copy** is True, then the memoryview will always expose a new copy of
        the data.

        In a C level implementation of the DataDescriptor interface, this
        returns a (void*) pointer to the data.
        """
        raise NotImplementedError

    def asbuflist(self, copy=False):
        """ Returns the contents of the buffer as a list of memoryviews.  If
        **copy** is False, then tries to return just views of data if possible.
        """
        raise NotImplementedError

    def asstream(self):
        """ Returns a Python iterable which returns **chunksize** elements
        at a time from the buffer.  This is identical to the iterable interface
        around Buffer objects.

        If **chunksize** is greater than 1, then returns a memoryview of the
        elements if they are contiguous, or a Tuple otherwise.
        """
        raise NotImplementedError

    def asstreamlist(self):
        """ Returns a list of Stream objects, which should be read sequentially
        (i.e. after exhausting the first stream, the second stream is read,
        etc.)
        """
        raise NotImplementedError

#------------------------------------------------------------------------
# Python Reference Implementations
#------------------------------------------------------------------------

class Buffer(object):
    """ Describes a region of memory. Implements the memoryview interface.
    """
    desctype = "buffer"

    length   = Int     # Total length, in bytes, of the buffer
    format   = Str     # Format of each elem, in struct module syntax
    shape    = Tuple    # Numpy-style shape tuple
    strides  = Tuple  #
    readonly = Bool(False)

    # TODO: Add support for event callbacks when certain ranges are
    # written
    #callbacks = Dict(Tuple, Function)

    def tobytes(self):
        pass

    def tolist(self):
        pass

class Stream(DataDescriptor):
    """ Describes a data item that must be retrieved by calling a function
    to load data into memory.  Represents a scan over data.  The returned
    data is a copy and should be considered to be owned by the caller.
    """
    desctype = "stream"

    length    = Int
    format    = Str
    chunksize = Int(1)  # optional "optimal" number of elements to read

    def read(self, nbytes):
        pass

    def move(self, dest, src, nbytes):
        pass


#------------------------------------------------------------------------
# Data descriptor implementations
#------------------------------------------------------------------------

class Chunk(object):

    pointer  = Int     # Data pointer to first element, Py_uintptr_t
    shape    = Tuple   # Numpy-style shape tuple
    strides  = Tuple   # Strides of the chunk
    itemsize = Int     # dtype itemsize
    readonly = Bool(False)

    def __init__(self, pointer, shape, strides, itemsize):
        super(Chunk, self).__init__()
        self.pointer = pointer
        self.shape = shape
        self.strides = strides
        self.itemsize = itemsize

class CArrayDataDescriptor(DataDescriptor):

    def __init__(self, id, nbytes, carray):
        super(CArrayDataDescriptor, self).__init__(id, nbytes)
        self.carray = carray
        self.itemsize = carray.itemsize

    def build_chunk(self, pointer, length):
        return Chunk(pointer, (length,), (self.itemsize,), self.itemsize)

    def asbuflist(self, copy=False):
        # TODO: incorporate shape in the chunks
        for chunk in self.carray.chunks:
            yield self.build_chunk(chunk.pointer, chunk.nbytes / self.itemsize)

        leftover_array = self.carray.leftover_array
        if leftover_array is not None:
            yield self.build_chunk(leftover_array.ctypes.data,
                                   leftover_array.shape[0])