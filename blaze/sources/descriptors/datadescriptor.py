from blaze import byteproto as proto
from . import lldescriptors, llindexers

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

    def __init__(self, id, nbytes, datashape):
        # XXX: whatever, just something unique for now
        self.id = id
        self.nbytes = nbytes
        self.datashape = datashape

    def __str__(self):
        return self.id

    def __repr__(self):
        return self.id

    #------------------------------------------------------------------------
    # Generic adapters
    #------------------------------------------------------------------------

    # From (roughly) least to most preferable

    def asstrided(self, copy=False):
        """ Returns the buffer as a memoryview. If **copy** is False, then the
        memoryview just references the underlying data if possible.  If
        **copy** is True, then the memoryview will always expose a new copy of
        the data.

        In a C level implementation of the DataDescriptor interface, this
        returns a (void*) pointer to the data.
        """
        raise NotImplementedError


    def asindex(self):
        """ Returns an indexer that can index the source with given
        coordinates
        """

    def as_tile_indexer(self, copy=False):
        """ Returns the contents of the buffer as an indexer returning
        N-dimensional memoryviews.

        A 1D chunk is a degenerate tile

        If **copy** is False, then tries to return just views of
        data if possible.
        """
        raise NotImplementedError

    def as_chunked_iterator(self, copy=False):
        """Return a ChunkIterator
        """
        raise NotImplementedError

    # NOTE: Buffered streams can be thought of as 1D tiles.
    # TODO: Remove stream interface and expose tiling properties in graph
    # TODO: metadata
    def asstream(self):
        """ Returns a Python iterable which returns **chunksize** elements
        at a time from the buffer.  This is identical to the iterable interface
        around Buffer objects.

        If **chunksize** is greater than 1, then returns a memoryview of the
        elements if they are contiguous, or a Tuple otherwise.
        """
        raise NotImplementedError

    def asstreamlist(self):
        """ Returns an iterable of Stream objects, which should be read sequentially
        (i.e. after exhausting the first stream, the second stream is read,
        etc.)
        """
        raise NotImplementedError

#------------------------------------------------------------------------
# Python Reference Implementations
#------------------------------------------------------------------------

# struct Buffer {
#     int length;
#     char* format;
#     int* shape;
#     int* strides;
#     int readonly;
# }

class Buffer(object):
    """ Describes a region of memory. Implements the memoryview interface.
    """
    desctype = "buffer"

    # TODO: Add support for event callbacks when certain ranges are
    # written
    # TODO: Why do we want this? ~Stephen
    #callbacks = Dict(Tuple, Function)

    def tobytes(self):
        pass

    def tolist(self):
        pass

# struct Stream {
#     int length;
#     char* format;
#     int chunksize;
#     int (*next)();
# }

class Stream(DataDescriptor):
    """ Describes a data item that must be retrieved by calling a function
    to load data into memory.  Represents a scan over data.  The returned
    data is a copy and should be considered to be owned by the caller.
    """
    desctype = "stream"

#------------------------------------------------------------------------
# Data descriptor implementations
#------------------------------------------------------------------------

# struct Chunk {
#     int pointer;
#     int* shape;
#     int* strides;
#     int itemsize;
#     int readonly;
# }

class SqlDataDescriptor(DataDescriptor):

    def __init__(self, id, conn, query):
        self.id = id
        self.conn = conn
        self.query = query

    def asbuffer(self, copy=False):
        self.conn.execute(self.query)
        return self.conn.fetchone()

class CArrayDataDescriptor(DataDescriptor):

    def __init__(self, id, nbytes, datashape, carray):
        super(CArrayDataDescriptor, self).__init__(id, nbytes, datashape)
        self.carray = carray
        self.itemsize = carray.itemsize

    def as_chunked_iterator(self, copy=False):
        """Return a ChunkIterator
        """
        return llindexers.CArrayChunkIterator(self.carray, self.datashape)
