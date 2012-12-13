from lldescriptors cimport *
from cpython cimport *

class Chunk(object):

    def __init__(self, pointer, shape, strides, itemsize):
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

        # main chunks
        for chunk in self.carray.chunks:
            yield self.build_chunk(chunk.pointer, chunk.nbytes / self.itemsize)

        # main leftovers
        leftover_array = self.carray.leftover_array
        if leftover_array is not None:
            yield self.build_chunk(leftover_array.ctypes.data,
                                   leftover_array.shape[0])

cdef class CArrayTileIndexer(object):
    def __cinit__(self, datasource, datashape, *args, **kwargs):
        self.datasource = datasource
        self.datashape = datashape
        self.indexer.index = carray_tile_read
        self.indexer.commit = carray_tile_commit

cdef void carray_tile_read(CTileIndexer *info, Py_ssize_t *indices,
                           CTile *out_tile):
    carray = <object> <PyObject *> info.meta.source


cdef void carray_tile_commit(CTileIndexer *info, Py_ssize_t *indices,
                             CTile *in_tile):
    pass