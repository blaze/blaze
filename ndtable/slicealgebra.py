"""
The algebra of slice indexer objects down into memory objects. Since the
Blaze model is based on recursive structuring of data regions this may
involve several levels of calls to get to "the bottom turtle".

Ostensibly a "toy" implementation of the core methods of Numpy in pure
Python to ensure that our generalization contains Numpy proper.
"""

# ==================================================================
# Numpy
# ==================================================================

#        buffer start       dot product          cast into native
#            |                   |                      |
# ptr = (char *)buf + ( indices dot strides ) = *((typeof(item) *)ptr);

import numpy as np
from struct import unpack
from itertools import izip

# void *
# PyArray_GetPtr(PyArrayObject *obj, npy_intp* ind)
# {
#     int n = obj->nd;
#     npy_intp *strides = obj->strides;
#     char *dptr = obj->data;
#
#     while (n--) {
#         dptr += (*strides++) * (*ind++);
#     }
#     return (void *)dptr;
# }

def numpy_get(na, indexer):

    n       = na.ndim
    strides = na.strides
    data    = na.data
    size    = na.dtype.itemsize
    kind    = na.dtype.char
    addr    = 0

    for i in reversed(xrange(n)):
        addr += strides[i] * indexer[i]

    return unpack(kind, data[addr:addr+size])[0]

# void *get_item_pointer(int ndim, void *buf, Py_ssize_t *strides,
#                        Py_ssize_t *suboffsets, Py_ssize_t *indices) {
#     char *pointer = (char*)buf;
#     int i;
#     for (i = 0; i < ndim; i++) {
#         pointer += strides[i] * indices[i];
#         if (suboffsets[i] >=0 ) {
#             pointer = *((char**)pointer) + suboffsets[i];
#         }
#     }
#     return (void*)pointer;
# }

# Pil is an extension of this but with suboffsets.
def pil_get(na, indexer):
    pass

# array_iter_base_init(PyArrayIterObject *it, PyArrayObject *ao)
# {
#    int nd, i;
#
#    nd = ao->nd;
#   it->ao = ao;
#   it->size = PyArray_SIZE(ao);
#   it->nd_m1 = nd - 1;
#   it->factors[nd-1] = 1;
#   for (i = 0; i < nd; i++) {
#       it->dims_m1[i] = ao->dimensions[i] - 1;
#       it->strides[i] = ao->strides[i];
#       it->backstrides[i] = it->strides[i] * it->dims_m1[i];
#       if (i > 0) {
#           it->factors[nd-i-1] = it->factors[nd-i] * ao->dimensions[nd-i];
#       }
#       it->bounds[i][0] = 0;
#       it->bounds[i][1] = ao->dimensions[i] - 1;
#       it->limits[i][0] = 0;
#       it->limits[i][1] = ao->dimensions[i] - 1;
#       it->limits_sizes[i] = it->limits[i][1] - it->limits[i][0] + 1;
#   }
# }


#define PyArray_ITER_NEXT(it) {                                            \
#         _PyAIT(it)->index++;                                               \
#         if (_PyAIT(it)->nd_m1 == 0) {                                      \
#                 _PyArray_ITER_NEXT1(_PyAIT(it));                           \
#         }                                                                  \
#         else if (_PyAIT(it)->contiguous)                                   \
#                 _PyAIT(it)->dataptr += _PyAIT(it)->ao->descr->elsize;      \
#         else if (_PyAIT(it)->nd_m1 == 1) {                                 \
#                 _PyArray_ITER_NEXT2(_PyAIT(it));                           \
#         }                                                                  \
#         else {                                                             \
#                 int __npy_i;                                               \
#                 for (__npy_i=_PyAIT(it)->nd_m1; __npy_i >= 0; __npy_i--) { \
#                         if (_PyAIT(it)->coordinates[__npy_i] <             \
#                             _PyAIT(it)->dims_m1[__npy_i]) {                \
#                                 _PyAIT(it)->coordinates[__npy_i]++;        \
#                                 _PyAIT(it)->dataptr +=                     \
#                                         _PyAIT(it)->strides[__npy_i];      \
#                                 break;                                     \
#                         }                                                  \
#                         else {                                             \
#                                 _PyAIT(it)->coordinates[__npy_i] = 0;      \
#                                 _PyAIT(it)->dataptr -=                     \
#                                         _PyAIT(it)->backstrides[__npy_i];  \
#                         }                                                  \
#                 }                                                          \
#         }                                                                  \
# }

def numpy_iter(na):
    pass

def numpy_binary_loop(f, a, b):
    ai = numpy_iter(a)
    bi = numpy_iter(b)

    for i,j in izip(ai, bi):
        yield f(i,j)

def numpy_unary_loop(f, a):
    ai = numpy_iter(a)

    for i in ai:
        yield f(i)

# PyUFunc_FromFuncAndData(PyUFuncGenericFunction *func, void **data,
#                         char *types, int ntypes,
#                         int nin, int nout, int identity,
#                         char *name, char *doc, int check_return)

def numpy_ufunc(fn, data, types, ntypes, nin, nout):
    pass

# ==================================================================
# General Slice Algebra
# ==================================================================

# Numpy is a linear mapping between strides and indexers. In numpy the
# for example the datshape gives (2, 3, int32):

#   With n = 2
#   With stride[i] = shape[i+1] * ... *shape[n]

# Yields strides
#   (24, 8)

# The new general form is when our strides become symbolic entities in
# their own right and instead of being just multiplication, they take on
# a function corresponding to the datashape. I propose we call these the
# "traversers" of which strides are a subset.

#   Fixed     ~ (*) -- equiveleant to Numpy
#   Var       ~
#   TypeVar   ~
#   Enum(i)   ~ [ a, b, c, ...][i]
#   Record(k) ~ { 'a': ds1, 'b': ds2, ... }[k]

# With traverse[i]   = f( datashape[i], datashape[i+1], ... )
#      traverse[i+1] = f( datashape[i+2], datashape[i+1], ... )
#      ...
#      traverse[n]   = f( datashape[n] )

def blaze_get(na, indexer):
    pass

def blaze_iter():
    pass

def blaze_binary_loop(f, a, b):
    ai = blaze_iter(a)
    bi = blaze_iter(b)

    for i,j in izip(ai, bi):
        yield f(i,j)

def blaze_unary_loop(f, a):
    ai = blaze_iter(a)

    for i in ai:
        yield f(i)

def blaze_ufunc(fn, data, types, ntypes, nin, nout):
    pass
