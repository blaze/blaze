cimport cpython
cimport libc.stdlib
cimport numpy as cnp
from cpython cimport PyObject, Py_INCREF

import numpy as np

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(
            PyObject *subtype, object descr, int nd,
            cnp.npy_intp* dims, cnp.npy_intp* strides,
            void* data, int flags, PyObject *obj)

cnp.import_array()

cdef build_fake_array(dtype):
    cdef cnp.npy_intp shape = 0
    cdef cnp.npy_intp strides = dtype.itemsize
    cdef int flags = cnp.NPY_C_CONTIGUOUS | cnp.NPY_WRITEABLE

    # PyArray_NewFromDescr will steal our reference
    Py_INCREF(dtype)

    array = PyArray_NewFromDescr(
        <PyObject *> np.ndarray,        # subtype
        dtype,                          # descr
        1,                              # ndim
        &shape,                         # shape
        &strides,                       # strides
        <void *> 1,                     # data
        flags,                          # flags
        NULL,                           # obj
    )

    return array

cdef class Executor(object):
    cdef execute(self, void **data_pointers, void *out, size_t size):
        "Execute a kernel over the data"

cdef class ElementwiseLLVMExecutor(Executor):
    cdef object ufunc, result_dtype
    cdef list operands, dtypes
    cdef object lhs_array
    cdef void *lhs_data

    def __init__(self, ufunc, dtypes, result_dtype):
        self.ufunc = ufunc
        self.result_dtype = result_dtype

        self.operands = []
        for dtype in dtypes:
            array = build_fake_array(dtype)
            self.operands.append(array)

        self.lhs_array = build_fake_array(result_dtype)

    cdef allocate_empty_lhs(self, size_t size):
        cdef size_t itemsize = self.result_dtype.itemsize
        self.lhs_data = libc.stdlib.malloc(itemsize * size)
        if self.lhs_data == NULL:
            raise MemoryError()

    cdef execute(self, void **data_pointers, void *out, size_t size):
        """
        Execute a kernel over the data

        TODO: strides, reductions
        """
        cdef int i
        cdef size_t itemsize

        for i, operand in enumerate(self.operands):
            (<cnp.ndarray> self.operands[i]).data = <char *> data_pointers[i]

        if out == NULL:
            if self.lhs_data == NULL:
                self.allocate_empty_lhs(size)
            out = self.lhs_data

        (<cnp.ndarray> self.lhs_array).data = <char *> out

        self.ufunc(*self.operands, out=self.lhs_array)


def execution(Executor executor, operands):
    cdef void **data_pointers = stdlib.malloc(len(operands) * sizeof(void *))
    if data_pointers == NULL:
        raise MemoryError

    try:
        for operand in operands:
            operand.data
    finally:
        free(data_pointers)

