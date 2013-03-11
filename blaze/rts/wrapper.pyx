import cython
import numpy as np

from cpython cimport PyObject, Py_INCREF
from libc.stdlib cimport malloc, free

cimport numpy as np
cimport wrapper as _wrap

np.import_array()

ctypedef unsigned long long ptr_t

cdef extern from "Python.h":
    ctypedef unsigned int Py_uintptr_t

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(
            PyObject *subtype, object descr, int nd,
            np.npy_intp* dims, np.npy_intp* strides,
            void* data, int flags, PyObject *obj)

#------------------------------------------------------------------------
# Array View
#------------------------------------------------------------------------

def len_chunk(chunk):
    return cython.cdiv(chunk.nbytes, chunk.itemsize)

cdef array_container(dtype):
    cdef:
        np.npy_intp shape = 0
        np.npy_intp strides = 0
        int flags = np.NPY_WRITEABLE | np.NPY_C_CONTIGUOUS

    Py_INCREF(dtype)

    array = PyArray_NewFromDescr(
        <PyObject *> np.ndarray,        # subtype
        dtype,                          # descr
        1,                              # ndim
        &shape,                         # shape
        &strides,                       # strides
        <void *>1,                      # data
        flags,                          # flags
        NULL,                           # obj
    )

    return array

def view(chunk):
    cdef:
       size_t       itemsize
       Py_ssize_t   *strides
       int          chunkpointer
       char*        data
       np.ndarray  arr

    strides = <Py_ssize_t *>malloc(sizeof(Py_ssize_t))

    chunkpointer = int(chunk.pointer)
    num_elements = len_chunk(chunk)

    arr = array_container(chunk.dtype)
    arr.data = <char*>chunkpointer
    arr.shape[0] = num_elements
    arr.strides[0] = 4
    return arr

#------------------------------------------------------------------------
# Runtime Wrapper
#------------------------------------------------------------------------

cdef class Source(object):

    def __init__(self, operands):
        self.operands = []

        #for i, chunk in enumerate(operands):
            #array_container()

cdef class Sink(object):

    def __init__(self):
        pass

cdef class Runtime(object):
    cdef void *rts

    def __init__(self, ptr_t kernel, ptr_t args, int arglen):
        self.rts = _wrap.init_runtime(5)

    def join(self):
        _wrap.join_runtime(self.rts)
