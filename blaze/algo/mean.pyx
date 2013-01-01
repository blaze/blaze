import cython

import numpy as np
cimport numpy as np

from blaze import Table
from blaze.carray import carrayExtension as carray
from blaze.carray.carrayExtension cimport chunk


np.import_array()
nan = np.nan

#------------------------------------------------------------------------
# Columwise Mean
#------------------------------------------------------------------------

def mean(table, col):
    """ Columnwise out of core mean

    Parameters
    ----------
    table : Table
        A Blaze Table object
    col : str
        String indicating a column name.

    Returns
    -------
    out : float
        mean

    """
    cdef chunk chunk_
    cdef np.npy_intp nchunk, nchunks
    cdef np.float32_t result = 0
    cdef Py_ssize_t count = 0
    col = table.data.ca[col]

    leftover = (col.len % len(col.leftover_array)) * col.atomsize
    nchunks = col.nchunks

    for nchunk in range(0, nchunks):
      chunk_ = col.chunks[nchunk]

      if chunk_.isconstant:
        result += chunk_.constant * col.chunklen
      else:
        result += chunk_[:].sum(dtype=col.dtype)

      count += cython.cdiv(chunk_.nbytes, chunk_.atomsize)

    if col.leftovers:
      leftover = col.len - nchunks * col.chunklen
      result += col.leftover_array[:leftover].sum(dtype=col.dtype)
      count += leftover

    if count > 0:
        return np.float32(result / count)
    else:
        return np.float32(nan)

#------------------------------------------------------------------------
# NA Experiments
#------------------------------------------------------------------------

# Experimental, don't use!

@cython.boundscheck(False)
@cython.wraparound(False)
def mean_float(np.ndarray[np.float32_t, ndim=1] a, Py_ssize_t _count=0):
    cdef Py_ssize_t count = _count

    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = np.PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]

    with nogil:
        for i0 in range(n0):
            ai = a[i0]
            if ai == ai: # disregard nan
                asum += ai
                count += 1

    if count > 0:
        return np.float32(asum / count), count
    else: return np.float32(nan), count

@cython.boundscheck(False)
@cython.wraparound(False)
def mean_int(np.ndarray[np.int32_t, ndim=1] a, int lower, int upper, Py_ssize_t _count=0):
    cdef Py_ssize_t count = _count

    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t size
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = np.PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    size = n0

    with nogil:
        for i0 in range(n0):
            ai = a[i0]
            if lower < ai < upper:
                asum += ai
                count += 1

    if count > 0:
        return np.float32(asum / count), count
    else:
        return np.float32(nan), count
