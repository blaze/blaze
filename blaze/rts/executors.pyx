cimport cpython
from libc.stdlib cimport malloc, free
cimport numpy as cnp
from cpython cimport PyObject, Py_INCREF
from blaze.compile.llvm import numba_kernels

from blaze.desc cimport lldescriptors

#------------------------------------------------------------------------
# NumPy Utilities
#------------------------------------------------------------------------

import numba
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
    "Build a numpy array without data"
    cdef cnp.npy_intp shape = 0
    cdef cnp.npy_intp strides = 0       # dtype.itemsize
    cdef int flags = cnp.NPY_WRITEABLE  # | cnp.NPY_C_CONTIGUOUS

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

#------------------------------------------------------------------------
# Various forms of Executors
#------------------------------------------------------------------------

cdef class Executor(object):

    cdef public object strategy
    cdef public object operation # for debugging and printing

    def __init__(self, strategy, operation="<operation>"):
        # assert strategy in ('chunked', 'tiled', 'indexed')
        self.strategy = strategy
        self.operation = operation

    def __call__(self, operands, out_operand):
        "Execute a kernel over the data given the operands and the LHS"
        # print operands, out_operand
        method = getattr(self, "execute_%s" % self.strategy)
        return method(operands, out_operand)

    def execute_chunked(self, operands, out_operand):
        cdef int i
        cdef void **data_pointers
        cdef Py_ssize_t *strides

        cdef lldescriptors.Chunk chunk

        operands.append(out_operand)
        descriptors = [op.data.read_desc() for op in operands]

        nbytes_list = np.array([desc.nbytes for desc in descriptors])
        # assert np.all(nbytes_list == nbytes_list[0]), nbytes_list

        # TODO: stack-allocate MAX_OPERANDS entries
        data_pointers = <void **> malloc(len(operands) * sizeof(void *))
        strides = <Py_ssize_t *> malloc(len(operands) * sizeof(Py_ssize_t))

        try:
            if data_pointers == NULL or strides == NULL:
                raise MemoryError

            # TODO: match up chunks of different sizes (and recognize reductions)
            iterators = [desc.as_chunked_iterator() for desc in descriptors]
            assert len(iterators) >= 2, iterators

            for paired_chunks in zip(*iterators):
                # paired_chunks, lhs_chunk = paired_chunks[:-1], paired_chunks[-1]
                for i, chunk in enumerate(paired_chunks):
                    data_pointers[i] = chunk.chunk.data
                    strides[i] = chunk.chunk.stride

                self.execute_chunk(data_pointers, strides, chunk.chunk.size,
                                   paired_chunks)

                for it, chunk in zip(iterators, paired_chunks):
                    it.dispose(chunk)

                lhs_chunk = paired_chunks[-1]
                iterators[-1].commit(lhs_chunk)
        finally:
            free(data_pointers)
            free(strides)

        return out_operand

    cdef execute_chunk(self, void **data_pointers, Py_ssize_t *strides,
                             size_t size, tuple paired_chunks):
        raise NotImplementedError

    def __repr__(self):
        return "Executor(%s, %s)" % (self.strategy, self.operation)

#------------------------------------------------------------------------
# Numba UFunc Executors
#------------------------------------------------------------------------

cdef class ElementwiseLLVMExecutor(Executor):
    """
    Use a ufunc with a Numba kernel for element-wise execution.
    """

    cdef object ufunc, result_dtype
    cdef list operands, dtypes
    cdef object lhs_array

    def __init__(self, strategy, ufunc, dtypes, result_dtype, **kwargs):
        super(ElementwiseLLVMExecutor, self).__init__(strategy, **kwargs)
        self.ufunc = ufunc
        self.result_dtype = result_dtype

        self.operands = []
        for dtype in dtypes + [result_dtype]:
            array = build_fake_array(dtype)
            self.operands.append(array)

        self.lhs_array = array

    cdef execute_chunk(self, void **data_pointers, Py_ssize_t *strides,
                             size_t size, tuple paired_chunks):
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
            op.strides[0] = strides[i]
            # print hex(<Py_uintptr_t> op.data), size

        # print "lhs", hex(<Py_uintptr_t> op.data), size
        self.ufunc(*self.operands[:-1], out=self.lhs_array)

    def __repr__(self):
        return "LLVMExecutor(%s, %s)" % (self.strategy, self.operation)

#------------------------------------------------------------------------
# Numba Kernel Executors
#------------------------------------------------------------------------
# Kernels written entirely in Numba

cdef class NumbaFullReducingExecutor(Executor):
    """
    Perform full reductions, e.g. A.sum()

        numba_reducer: python numba scalar kernel
        numba_type: numba type representation of the element type (dtype)
    """

    cdef object reduce_kernel, numba_type

    def __init__(self, strategy, numba_reducer, numba_type, **kwargs):
        super(NumbaFullReducingExecutor, self).__init__(strategy, **kwargs)
        self.reduce_kernel = numba.autojit(numba_reducer)
        self.numba_type = numba_type

    cdef execute_chunk(self, void **data_pointers, Py_ssize_t *strides,
                             size_t size, tuple paired_chunks):
        # Use the RHS to determine the data size, the scalar LHS is used
        # only to reduce
        cdef lldescriptors.Chunk chunk = paired_chunks[0]

        numba_kernels.numba_full_reduce(
                <Py_uintptr_t> data_pointers,
                strides[0],
                chunk.chunk.size,
                self.reduce_kernel,
                self.numba_type,
                self.numba_type.pointer())
