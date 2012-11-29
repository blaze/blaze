import ctypes

from libc.math cimport sin
from libc.stdlib cimport free, malloc
from ctypes import c_int, c_void_p, c_double

cdef extern from "blosc.h" nogil:
    enum: BLOSC_MAX_OVERHEAD

    int blosc_compress(int clevel, int doshuffle, size_t typesize,
                       size_t nbytes, void *src, void *dest, size_t destsize)
    int blosc_decompress(void *src, void *dest, size_t destsize)
    void blosc_free_resources()
    int blosc_set_nthreads(int nthreads)

ctypedef void (*ufuncptr_t)(void** args, size_t* dimensions, size_t* steps,
    void* data)
ctypedef void* (*asbuffer_t)(int)

cdef class Datashape:
    cdef public int itemsize
    cdef public size_t* dimensions
    cdef public size_t* steps

    def __cinit__(self, itemsize):
        self.itemsize = itemsize

cdef class DataDescriptor:
    cdef public int nchunks
    cdef public int chunksize
    cdef asbuffer_t asbuffer

    def __cinit__(self, nchunks, chunksize, int asbuffer):
        self.nchunks   = nchunks
        self.chunksize = chunksize
        self.asbuffer  = <asbuffer_t>asbuffer

cdef map_dispatch(ufuncptr_t ufuncptr, DataDescriptor dd, Datashape ds):
    cdef char *buf = <char*>malloc(dd.chunksize)
    cdef char *tmp = <char*>malloc(dd.chunksize*dd.nchunks)

    for i in xrange(dd.nchunks):
        # Load
        buf = <char*>dd.asbuffer(i)
        # Execute
        ufuncptr(<void**>buf, ds.dimensions, ds.steps, tmp)
