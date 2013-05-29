########################################################################
#
#       License: BSD
#       Created: September 10, 2010
#       Author:  Francesc Alted - francesc@continuum.io
#
########################################################################

"""Top level functions and classes.
"""

import sys
import os, os.path
import glob
import itertools as it
import numpy as np
from blz_ext import barray
from blaze.blz.btable import btable
from bparams import bparams
import math


def open(rootdir, mode='a'):
    """
    open(rootdir, mode='a')

    Open a disk-based barray/btable.

    Parameters
    ----------
    rootdir : pathname (string)
        The directory hosting the barray/btable object.
    mode : the open mode (string)
        Specifies the mode in which the object is opened.  The supported
        values are:

          * 'r' for read-only
          * 'w' for emptying the previous underlying data
          * 'a' for allowing read/write on top of existing data

    Returns
    -------
    out : a barray/btable object or None (if not objects are found)

    """
    # First try with a barray
    obj = None
    try:
        obj = barray(rootdir=rootdir, mode=mode)
    except IOError:
        # Not a barray.  Now with a btable
        try:
            obj = btable(rootdir=rootdir, mode=mode)
        except IOError:
            # Not a btable
            pass
    return obj

def fromiter(iterable, dtype, count, **kwargs):
    """
    fromiter(iterable, dtype, count, **kwargs)

    Create a barray/btable from an `iterable` object.

    Parameters
    ----------
    iterable : iterable object
        An iterable object providing data for the barray.
    dtype : numpy.dtype instance
        Specifies the type of the outcome object.
    count : int
        The number of items to read from iterable. If set to -1, means that
        the iterable will be used until exhaustion (not recommended, see note
        below).
    kwargs : list of parameters or dictionary
        Any parameter supported by the barray/btable constructors.

    Returns
    -------
    out : a barray/btable object

    Notes
    -----
    Please specify `count` to both improve performance and to save memory.  It
    allows `fromiter` to avoid looping the iterable twice (which is slooow).
    It avoids memory leaks to happen too (which can be important for large
    iterables).
    
    """
    from btable import btable

    # Check for a true iterable
    if not hasattr(iterable, "next"):
        iterable = iter(iterable)

    # Try to guess the final length
    expected = count
    if count == -1:
        # Try to guess the size of the iterable length
        if hasattr(iterable, "__length_hint__"):
            count = iterable.__length_hint__()
            expected = count
        else:
            # No guess
            count = sys.maxint
            # If we do not have a hint on the iterable length then
            # create a couple of iterables and use the second when the
            # first one is exhausted (ValueError will be raised).
            iterable, iterable2 = it.tee(iterable)
            expected = 1000*1000   # 1 million elements

    # First, create the container
    expectedlen = kwargs.pop("expectedlen", expected)
    dtype = np.dtype(dtype)
    if dtype.kind == "V":
        # A btable
        obj = btable(np.array([], dtype=dtype),
                     expectedlen=expectedlen, **kwargs)
        chunklen = sum(obj.cols[name].chunklen
                       for name in obj.names) // len(obj.names)
    else:
        # A barray
        obj = barray(np.array([], dtype=dtype),
                     expectedlen=expectedlen, **kwargs)
        chunklen = obj.chunklen

    # Then fill it
    nread, blen = 0, 0
    while nread < count:
        if nread + chunklen > count:
            blen = count - nread
        else:
            blen = chunklen
        if count != sys.maxint:
            chunk = np.fromiter(iterable, dtype=dtype, count=blen)
        else:
            try:
                chunk = np.fromiter(iterable, dtype=dtype, count=blen)
            except ValueError:
                # Positionate in second iterable
                iter2 = it.islice(iterable2, nread, None, 1)
                # We are reaching the end, use second iterable now
                chunk = np.fromiter(iter2, dtype=dtype, count=-1)
        obj.append(chunk)
        nread += len(chunk)
        # Check the end of the iterable
        if len(chunk) < chunklen:
            break
    obj.flush()
    return obj

def fill(shape, dflt=None, dtype=np.float, **kwargs):
    """
    fill(shape, dtype=float, dflt=None, **kwargs)

    Return a new barray object of given shape and type, filled with `dflt`.

    Parameters
    ----------
    shape : int
        Shape of the new array, e.g., ``(2,3)``.
    dflt : Python or NumPy scalar
        The value to be used during the filling process.  If None, values are
        filled with zeros.  Also, the resulting barray will have this value as
        its `dflt` value.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    kwargs : list of parameters or dictionary
        Any parameter supported by the barray constructor.

    Returns
    -------
    out : barray
        Array filled with `dflt` values with the given shape and dtype.

    See Also
    --------
    ones, zeros

    """

    dtype = np.dtype(dtype)
    if type(shape) in (int, long, float):
        shape = (int(shape),)
    else:
        shape = tuple(shape)
        if len(shape) > 1:
            # Multidimensional shape.
            # The atom will have shape[1:] dims (+ the dtype dims).
            dtype = np.dtype((dtype.base, shape[1:]+dtype.shape))
    length = shape[0]

    # Create the container
    expectedlen = kwargs.pop("expectedlen", length)
    if dtype.kind == "V" and dtype.shape == ():
        raise ValueError, "fill does not support btables objects"
    obj = barray([], dtype=dtype, dflt=dflt, expectedlen=expectedlen,
                 **kwargs)
    chunklen = obj.chunklen

    # Then fill it
    # We need an array for the defaults so as to keep the atom info
    dflt = np.array(obj.dflt, dtype=dtype)
    # Making strides=(0,) below is a trick to create the array fast and
    # without memory consumption
    chunk = np.ndarray(length, dtype=dtype, buffer=dflt, strides=(0,))
    obj.append(chunk)
    obj.flush()
    return obj

def zeros(shape, dtype=np.float, **kwargs):
    """
    zeros(shape, dtype=float, **kwargs)

    Return a new barray object of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int
        Shape of the new array, e.g., ``(2,3)``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    kwargs : list of parameters or dictionary
        Any parameter supported by the barray constructor.

    Returns
    -------
    out : barray
        Array of zeros with the given shape and dtype.

    See Also
    --------
    fill, ones

    """
    dtype = np.dtype(dtype)
    return fill(shape=shape, dflt=np.zeros((), dtype), dtype=dtype, **kwargs)

def ones(shape, dtype=np.float, **kwargs):
    """
    ones(shape, dtype=float, **kwargs)

    Return a new barray object of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int
        Shape of the new array, e.g., ``(2,3)``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    kwargs : list of parameters or dictionary
        Any parameter supported by the barray constructor.

    Returns
    -------
    out : barray
        Array of ones with the given shape and dtype.

    See Also
    --------
    fill, zeros

    """
    dtype = np.dtype(dtype)
    return fill(shape=shape, dflt=np.ones((), dtype), dtype=dtype, **kwargs)

def arange(start=None, stop=None, step=None, dtype=None, **kwargs):
    """
    arange([start,] stop[, step,], dtype=None, **kwargs)

    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
    but returns a barray rather than a list.

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval.  The interval does not include this value.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified, `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    kwargs : list of parameters or dictionary
        Any parameter supported by the barray constructor.

    Returns
    -------
    out : barray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.

    """

    # Check start, stop, step values
    if (start, stop) == (None, None):
        raise ValueError, "You must pass a `stop` value at least."
    elif stop is None:
        start, stop = 0, start
    elif start is None:
        start, stop = 0, stop
    if step is None:
        step = 1

    # Guess the dtype
    if dtype is None:
        if type(stop) in (int, long):
            dtype = np.dtype(np.int_)
    dtype = np.dtype(dtype)
    stop = int(stop)

    # Create the container
    expectedlen = kwargs.pop("expectedlen", stop)
    if dtype.kind == "V":
        raise ValueError, "arange does not support btables yet."
    else:
        obj = barray(np.array([], dtype=dtype),
                     expectedlen=expectedlen,
                     **kwargs)
        chunklen = obj.chunklen

    # Then fill it
    incr = chunklen * step        # the increment for each chunk
    incr += step - (incr % step)  # make it match step boundary
    bstart, bstop = start, start + incr
    while bstart < stop:
        if bstop > stop:
            bstop = stop
        chunk = np.arange(bstart, bstop, step, dtype=dtype)
        obj.append(chunk)
        bstart = bstop
        bstop += incr
    obj.flush()
    return obj

def walk(dir, classname=None, mode='a'):
    """walk(dir, classname=None, mode='a')

    Recursively iterate over barray/btable objects hanging from `dir`.

    Parameters
    ----------
    dir : string
        The directory from which the listing starts.
    classname : string
        If specified, only object of this class are returned.  The values
        supported are 'barray' and 'btable'.
    mode : string
        The mode in which the object should be opened.

    Returns
    -------
    out : iterator
        Iterator over the objects found.

    """

    # First, iterate over the barray objects in current dir
    names = os.path.join(dir, '*')
    dirs = []
    for node in glob.glob(names):
        if os.path.isdir(node):
            try:
                obj = barray(rootdir=node, mode=mode)
            except:
                try:
                    obj = btable(rootdir=node, mode=mode)
                except:
                    obj = None
                    dirs.append(node)
            if obj:
                if classname:
                    if obj.__class__.__name__ == classname:
                        yield obj
                else:
                    yield obj

    # Then recurse into the true directories
    for dir_ in dirs:
        for node in walk(dir_, classname, mode):
            yield node


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 78
