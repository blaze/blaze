"""
Low-level data descriptors accessible from C, Cython and Python (and numba
in the future).
"""

#------------------------------------------------------------------------
# See lldescriptors.pxd for type definitions
#------------------------------------------------------------------------

cdef class lldatadesc(object):

    def __init__(self, data_obj, datashape):
        self.data_obj = data_obj
        self.datashape = datashape

cdef class ChunkIterator(lldatadesc):

    def __cinit__(self, data_obj, datashape, *args, **kwargs):
        super(ChunkIterator, self).__init__(data_obj, datashape)
        self.iterator.meta.source = <void *> data_obj
        self.iterator.meta.datashape = <void *> datashape

    def __iter__(self):
        cdef Chunk chunk = Chunk()
        cdef CChunk cchunk

        while True:
            self.iterator.next(&self.iterator, &cchunk)
            if cchunk.data == NULL:
                break

            chunk.chunk = cchunk
            yield chunk

cdef class Tile(object):
    """
    Tiles exposed to Python
    """

    def __init__(self, Py_uintptr_t data, tuple shape):
        cdef int i

        self.tile.ndim = len(shape)
        for i in range(len(shape)):
            self.tile.shape[i] = shape[i]
