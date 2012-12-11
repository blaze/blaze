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

#------------------------------------------------------------------------
# NumPy
#------------------------------------------------------------------------

cdef class ElementwiseNumpyExecutor(Executor):
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
        cdef cnp.ndarray op

        for i, operand in enumerate(self.operands):
            op = self.operands[i]
            op.data = <char *> data_pointers[i]
            op.shape[0] = size

        if out == NULL:
            raise NotImplementedError
            #if self.lhs_data == NULL:
            #    self.allocate_empty_lhs(size)
            #out = self.lhs_data

        op = self.lhs_array
        op.data = <char *> out
        op.shape[0] = size

        #print self.operands
        #print self.lhs_array
        self.ufunc(*self.operands, out=self.lhs_array)

#------------------------------------------------------------------------
# Numba
#------------------------------------------------------------------------

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
        cdef cnp.ndarray op

        for i, operand in enumerate(self.operands):
            op = self.operands[i]
            op.data = <char *> data_pointers[i]
            op.shape[0] = size

        if out == NULL:
            raise NotImplementedError
            #if self.lhs_data == NULL:
            #    self.allocate_empty_lhs(size)
            #out = self.lhs_data

        op = self.lhs_array
        op.data = <char *> out
        op.shape[0] = size

        #print self.operands
        #print self.lhs_array
        self.ufunc(*self.operands, out=self.lhs_array)

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------

# This is the toplevel command that the RTS will specialize to when
# we're dealing with ufuncs over operands that look like CArray objects

# TODO: one that doesn't assume chunking for NumPy
# Note Mark: why? a length-one loop isn't going to add any overhead

# TODO: much later a loop that deals with some of the more
# general datashapes

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
    descriptors = [op.data.read_desc() for op in operands]

    nbtyes = descriptors[0].nbytes
    assert all(desc.nbytes == nbtyes for desc in descriptors)

    data_pointers = <void **> malloc(len(operands) * sizeof(void *))
    if data_pointers == NULL:
        raise MemoryError

    try:
        # TODO: assert the chunks have equal sizes
        for paired_chunks in zip(*[desc.asbuflist() for desc in descriptors]):
            paired_chunks, lhs_chunk = paired_chunks[:-1], paired_chunks[-1]
            for i, chunk in enumerate(paired_chunks):
                #print hex(chunk.pointer)
                data_pointers[i] = <void *> <Py_uintptr_t> chunk.pointer

            #print hex(lhs_chunk.pointer)
            #print lhs_chunk.shape[0]
            lhs_data = <void *> <Py_uintptr_t> lhs_chunk.pointer
            executor.execute(data_pointers, lhs_data, lhs_chunk.shape[0])
    finally:
        free(data_pointers)
