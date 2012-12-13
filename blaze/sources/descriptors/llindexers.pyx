from lldescriptors cimport *
from cpython cimport *

from lldescriptors import *

from blaze.carray import carrayExtension as carray

cdef class CArrayChunkIterator(ChunkIterator):
    def __cinit__(self, data_obj, datashape, *args, **kwargs):
        super(CArrayChunkIterator, self).__init__(data_obj, datashape)
        self.iterator.next = carray_chunk_next
        self.iterator.commit = carray_chunk_commit

    def __iter__(self):
        cdef Chunk chunk = Chunk()
        cdef CChunk cchunk

        while True:
            self.iterator.next(&self.iterator, &cchunk)
            if cchunk.data == NULL:
                break

            chunk.chunk = cchunk
            yield chunk


cdef void carray_chunk_next(CChunkIterator *info, CChunk *chunk):
    cdef Py_uintptr_t data

    carray = <object> <PyObject *> info.meta.source
    if info.cur_chunk_idx < carray.nchunks:
        carray_chunk = carray.chunks[info.cur_chunk_idx]

        # decompress chunk
        arr = carray_chunk[:]
    elif info.cur_chunk_idx == carray.nchunks:
        arr = carray.leftover_array
    else:
        chunk.data = NULL
        # chunk.size = 0
        return

    chunk.data = <void *> <Py_uintptr_t> arr.ctypes.data
    chunk.size = arr.shape[0]
    chunk.stride = arr.strides[0]
    chunk.chunk_index = info.cur_chunk_idx

    # Keep decompressed memory alive
    Py_INCREF(arr)
    chunk.obj = <PyObject *> arr
    chunk.extra = <void *> carray_chunk

    info.cur_chunk_idx += 1

cdef void carray_chunk_commit(CChunkIterator *info, CChunk *chunk):
    carray_obj = <object> <PyObject *> info.meta.source
    if chunk.chunk_index < carray_obj.nchunks:
        # compress chunk and replace previous chunk
        carray_chunk = <object> chunk.extra
        arr = <object> chunk.obj
        carray.chunks[chunk.chunk_index] = carray.chunk(arr, arr.dtype,
                                                        carray_obj.cparams)

    carray_chunk_dispose(info, chunk)

cdef void carray_chunk_dispose(CChunkIterator *info, CChunk *chunk):
    # Decref previously set live object
    Py_XDECREF(chunk.obj)
    chunk.obj = NULL