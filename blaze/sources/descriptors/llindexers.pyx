from lldescriptors cimport *
from cpython cimport *

#from lldescriptors import *

from blaze.carray import carrayExtension as carray


cdef chunk_next_generic(CChunkIterator *info, CChunk *chunk, arr, keep_alive):
    """
    Fill out the chunk info given a numpy array
    """
    chunk.data = <void *> <Py_uintptr_t> arr.ctypes.data
    if arr.ndim > 0:
        chunk.size = arr.shape[0]
        chunk.stride = arr.strides[0]
    else:
        chunk.size = 1
        chunk.stride = 0

    chunk.chunk_index = info.cur_chunk_idx

    if keep_alive:
        # Keep chunk source alive (e.g. decompressed memory)
        # The dispose and commit functions must ensure this object gets
        # decreffed!
        Py_INCREF(arr)
        chunk.obj = <PyObject *> arr

    info.cur_chunk_idx += 1

cdef int done(CChunk *chunk):
    chunk.data = NULL
    return 0

#------------------------------------------------------------------------
# carray chunk iterators and indexers
#------------------------------------------------------------------------

cdef class CArrayChunkIterator(ChunkIterator):

    def __cinit__(self, data_obj, datashape, *args, **kwargs):
        super(CArrayChunkIterator, self).__init__(data_obj, datashape)
        self.iterator.next = carray_chunk_next
        self.iterator.commit = carray_chunk_commit
        self.iterator.dispose = carray_chunk_dispose

cdef int carray_chunk_next(CChunkIterator *info, CChunk *chunk) except -1:
    cdef Py_uintptr_t data

    carray = <object> <PyObject *> info.meta.source
    if info.cur_chunk_idx < carray.nchunks:
        carray_chunk = carray.chunks[info.cur_chunk_idx]

        # decompress chunk
        arr = carray_chunk[:]
        chunk.extra = <void *> carray_chunk
    elif info.cur_chunk_idx == carray.nchunks:
        arr = carray.leftover_array
    else:
        return done(chunk)

    chunk_next_generic(info, chunk, arr, True)
    return 0

cdef int carray_chunk_commit(CChunkIterator *info, CChunk *chunk) except -1:
    carray_obj = <object> <PyObject *> info.meta.source
    if chunk.chunk_index < carray_obj.nchunks:
        # compress chunk and replace previous chunk
        carray_chunk = <object> chunk.extra
        arr = <object> chunk.obj
        carray.chunks[chunk.chunk_index] = carray.chunk(arr, arr.dtype,
                                                        carray_obj.cparams)

    carray_chunk_dispose(info, chunk)
    return 0

cdef int carray_chunk_dispose(CChunkIterator *info, CChunk *chunk) except -1:
    # Decref previously set live object
    Py_XDECREF(chunk.obj)
    chunk.obj = NULL
    return 0

#------------------------------------------------------------------------
# NumPy chunk iterators and indexers
#------------------------------------------------------------------------

cdef class NumPyChunkIterator(ChunkIterator):

    def __cinit__(self, data_obj, datashape, *args, **kwargs):
        super(NumPyChunkIterator, self).__init__(data_obj, datashape)
        self.iterator.next = numpy_chunk_next
        self.iterator.commit = NULL
        self.iterator.dispose = NULL


cdef int numpy_chunk_next(CChunkIterator *info, CChunk *chunk) except -1:
    if info.cur_chunk_idx > 0:
        return done(chunk)

    arr = <object> <PyObject *> info.meta.source
    chunk_next_generic(info, chunk, arr, False)
    return 0
