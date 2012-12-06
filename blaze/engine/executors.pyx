cimport cpython
from libc.stdlib cimport malloc, free
cimport numpy as cnp
from cpython cimport PyObject, Py_INCREF

import numpy as np

cdef extern from "Python.h":
    ctypedef unsigned int Py_uintptr_t

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


def execute(Executor executor, operands, out_operand):
    """
    TODO: Implement a real algorithm that:

        1) Schedules chunks in a sensible manner
        2) Intersects differently shaped regions
        3) Takes into account dimensionalities > 1
    """
    cdef int i
    cdef void **data_pointers
    cdef void *lhs_data

    operands.append(out_operand)
    descriptors = [op.read_desc() for op in operands]

    nbtyes = descriptors[0].nbytes
    assert all(desc.nbytes == nbtyes for desc in descriptors)

    data_pointers = <void **> malloc(len(operands) * sizeof(void *))
    if data_pointers == NULL:
        raise MemoryError

    try:
        # TODO: assert the chunks have equal sizes
        for paired_chunks in zip(desc.asbuflist() for desc in descriptors):
            lhs_chunk = paired_chunks.pop()
            for i, chunk in enumerate(paired_chunks):
                data_pointers[i] = <void *> <Py_uintptr_t> chunk.pointer

            lhs_data = <void *> <Py_uintptr_t> lhs_chunk.pointer
            executor.execute(data_pointers, lhs_data, lhs_chunk.shape[0])
    finally:
        free(data_pointers)
