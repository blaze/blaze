import cython
import numpy as np
cimport numpy as np

from blaze import Table
from blaze.carray import carrayExtension as carray
from blaze.carray.carrayExtension cimport chunk

np.import_array()
nan = np.nan

# just using this to debug the loops to make sure we only do n
# reads from disk where n = nchunks.
def generic1d_loop(table, label):
    cdef:
        chunk chunk_

    col = table.data.ca[label]

    nchunks = col.nchunks

    for nchunk from 0 <= nchunk < nchunks:
        chunk_ = col.chunks[nchunk]
        size = cython.cdiv(chunk_.nbytes, chunk_.atomsize)

        if chunk_.isconstant:
            arr = chunk_.constant * col.chunklen
            print np.array([chunk._constant] * col.chunklen)
            # logic
        else:
            arr = chunk_[:]
            print arr
            # logic

    if col.leftovers:
        leftover = col.len - nchunks * col.chunklen
        arr = col.leftover_array[:leftover]
        print arr
        # logic

#------------------------------------------------------------------------
# Columwise Standard Deviation
#------------------------------------------------------------------------

# Calculate the sum and sum of squares in one pass
@cython.boundscheck(False)
@cython.wraparound(False)
cdef sqsum(np.ndarray[np.int32_t, ndim=1] a):
    cdef:
        Py_ssize_t length = a.shape[0]

        np.int64_t ai    = 0
        np.int64_t asum  = 0
        np.int64_t assum = 0

    for i from 0 <= i < length:
        ai = a[i]
        asum  += ai
        assum += ai * ai

    return asum, assum

def std(table, label):
    """ Columnwise out of core standard devaiation

    Parameters
    ----------
    table : Table
        A Blaze Table object
    col : str
        String indicating a column name.

    Returns
    -------
    out : float
        standard deviation

    """
    cdef:
        chunk chunk_
        Py_ssize_t nchunk, nchunks

        Py_ssize_t count = 0
        np.float64_t asum   = 0
        np.float64_t asumsq = 0
        np.float64_t amean  = 0

    col = table.data.ca[label]
    nchunks = col.nchunks
    count = col.len

    for nchunk from 0 <= nchunk < nchunks:
        chunk_ = col.chunks[nchunk]

        if chunk_.isconstant:
            it = chunk_.constant * col.chunklen
            asum += it
            asumsq += (it ** 2) * col.chunklen
        else:
            _asum, _assum = sqsum(chunk_[:])
            asum   += _asum
            asumsq += _assum

    if col.leftovers:
        leftover = col.len - nchunks * col.chunklen
        leftover_arr = col.leftover_array[:leftover]

        _asum, _assum = sqsum(leftover_arr)
        asum   += _asum
        asumsq += _assum

    if count > 0:
        amean = cython.cdiv(asum, count)
        # there is probably a more numerically stable version of this
        # but NumPy's implementation is really naive as well so whatever
        return np.float64(np.sqrt((asumsq / count) - (amean * amean)))
    else:
        return np.float64(nan)

#------------------------------------------------------------------------
# Columwise Mean
#------------------------------------------------------------------------

def mean(table, label):
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

    cdef:
        chunk chunk_
        Py_ssize_t nchunk, nchunks

        Py_ssize_t count = 0
        np.float64_t asum   = 0
        np.float64_t amean  = 0


    col = table.data.ca[label]
    nchunks = col.nchunks
    count = col.len

    for nchunk from 0 <= nchunk < nchunks:
        chunk_ = col.chunks[nchunk]

        if chunk_.isconstant:
            asum += chunk_.constant * col.chunklen
        else:
            asum += chunk_[:].sum(dtype=col.dtype)

    if col.leftovers:
        leftover = col.len - nchunks * col.chunklen
        asum += col.leftover_array[:leftover].sum(dtype=col.dtype)

    if count > 0:
        return np.float64(asum / count)
    else:
        return np.float64(nan)

#------------------------------------------------------------------------
# Columwise Lstsq
#------------------------------------------------------------------------

def lstsq(table, label):
    pass
