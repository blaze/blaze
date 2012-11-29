"""
A generalization of NumPy strides.

The "algebra" of slices maps indexer objects down into memory objects.
Since the Blaze model is based on recursive structuring of data regions
this may involve several levels of calls to get to "the bottom turtle".

The top is ostensibly a "toy" implementation of the core methods of
NumPy in pure Python to ensure that our generalization contains NumPy's
linear formula-based index.
"""

import numpy as np
from struct import unpack
from functools import partial
from operator import add, mul

# ==================================================================
# Math
# ==================================================================

# foldl :: (a -> a -> a) -> a -> [a] -> a
def foldl(op, origin, xs):
    """
     if xs = [a,b,c]
        origin = o
     then
       foldl f origin xs =
       (f (f o a) b) c
    """
    r = origin
    for x in xs:
        r = op(r, x)
    return r

# scanr :: (a -> b -> b) -> b -> [a] -> [b]
def scanr(op, origin, xs):
    """
    scanr f o [a,b,c] =
        [f a (f b (f c o)),f b (f c o),f c o,o]

    This yields precisely the classical Numpy formula for strides
    if op is mutliplication and take the tail.
    """
    i = origin
    r = [i]
    for x in reversed(xs):
        i = op(i,x)
        r.insert(0, i)
    return r

# zipwith :: (a -> b -> c) -> [a] -> [b] -> [c]
def zipwith(op, xs, ys):
    """
    zipwith g [a,b,c] [x,y,z] =
        [g a x, g b y, g c z]
    """
    return map(op, xs, ys)

def generalized_dot(f, g, o1, o2, xs, ys):
    return o1 + foldl(f, o2, zipwith(g, xs, ys))

dot = partial(generalized_dot, add, mul, 0, 0)

# ==================================================================
# Numpy
# ==================================================================

#        buffer start       dot product          cast into native
#            |                   |                      |
# ptr = (char *)buf + ( indices dot strides ) = *((typeof(item) *)ptr);

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

def numpy_strides(na):
    nd    = na.ndim
    shape = na.shape
    size  = na.dtype.itemsize

    # For C order
    return scanr(mul, size, shape[1:])

    # For Fortran order
    #return scanl(mul, size, shape[:-1])

def numpy_get(na, indexer):

    nd      = na.ndim
    data    = na.data
    size    = na.dtype.itemsize
    kind    = na.dtype.char
    addr    = 0

    strides = numpy_strides(na)
    addr = dot(strides, indexer)

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


#  define PyArray_ITER_NEXT(it) {                                            \
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

if __name__ == '__main__':
    a,b = [1,2,3], [4,5,6]
    c = np.ones((4,4,4))

    assert dot(a,b) == np.dot(a,b)

    assert numpy_strides(c) == list(c.strides)
    assert numpy_get(c, [0,0,1]) == c[0][0][1]
    assert numpy_get(c, [0,1,0]) == c[0][1][0]
    assert numpy_get(c, [1,0,0]) == c[1][0][0]
