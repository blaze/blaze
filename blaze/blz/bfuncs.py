########################################################################
#
#       License: BSD
#       Created: September 10, 2010
#       Author:  Francesc Alted - francesc@continuum.io
#
########################################################################

from __future__ import absolute_import

"""Top level functions and classes.
"""

import sys
import os, os.path
import glob
import itertools as it
import numpy as np
import math
from .blz_ext import barray
from .btable import btable
from .bparams import bparams
from ..py3help import xrange, _inttypes

_inttypes += (np.integer,)

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
    out : a barray/btable object

    """
    # Use the existence of __rootdirs__ to
    # distinguish between btable and barray
    if os.path.exists(os.path.join(rootdir, '__rootdirs__')):
        obj = btable(rootdir=rootdir, mode=mode)
    else:
        obj = barray(rootdir=rootdir, mode=mode)
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
    _MAXINT_SIGNAL = 2**64

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
            count = _MAXINT_SIGNAL
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
        if count != _MAXINT_SIGNAL:
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
    if type(shape) in _inttypes + (float,):
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
        raise ValueError("fill does not support btables objects")
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
        raise ValueError("You must pass a `stop` value at least.")
    elif stop is None:
        start, stop = 0, start
    elif start is None:
        start, stop = 0, stop
    if step is None:
        step = 1

    # Guess the dtype
    if dtype is None:
        if type(stop) in _inttypes:
            dtype = np.dtype(np.int_)
    dtype = np.dtype(dtype)
    stop = int(stop)

    # Create the container
    expectedlen = kwargs.pop("expectedlen", stop)
    if dtype.kind == "V":
        raise ValueError("arange does not support btables yet.")
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


def iterblocks(bobj, blen=None, start=0, stop=None):
    """iterblocks(blen=None, start=0, stop=None)

    Iterate over a `bobj` (barray/btable) in blocks of size `blen`.

    Parameters
    ----------
    bobj : barray/btable object
        The BLZ array to be iterated over.
    blen : int
        The length of the block that is returned.  The default is the
        chunklen, or for a btable, the minimum of the different column
        chunklens.
    start : int
        Where the iterator starts.  The default is to start at the beginning.
    stop : int
        Where the iterator stops. The default is to stop at the end.

    Returns
    -------
    out : iterable
        This iterable returns buffers as NumPy arays of homogeneous or
        structured types, depending on whether `bobj` is a barray or a
        btable object.

    See Also
    --------
    whereblocks

    """

    if stop is None:
        stop = len(bobj)
    if isinstance(bobj, btable):
        # A btable object
        if blen is None:
            # Get the minimum chunklen for every column
            blen = min(bobj[col].chunklen for col in bobj.cols)
        # Create intermediate buffers for columns in a dictarray
        # (it is important that columns are contiguous)
        cbufs = {}
        for name in bobj.names:
            cbufs[name] = np.empty(blen, dtype=bobj[name].dtype)
        for i in xrange(start, stop, blen):
            buf = np.empty(blen, dtype=bobj.dtype)
            # Populate the column buffers and assign to the final buffer
            for name in bobj.names:
                bobj[name]._getrange(i, blen, cbufs[name])
                buf[name][:] = cbufs[name]
            if i + blen > stop:
                buf = buf[:stop - i]
            yield buf
    else:
        # A barray object
        if blen is None:
            blen = bobj.chunklen
        for i in xrange(start, stop, blen):
            buf = np.empty(blen, dtype=bobj.dtype)
            bobj._getrange(i, blen, buf)
            if i + blen > stop:
                buf = buf[:stop - i]
            yield buf


def whereblocks(table, expression, blen=None, outfields=None, limit=None,
                skip=0):
    """
    whereblocks(table, expression, blen=None, outfields=None, limit=None, skip=0)

    Iterate over the rows that fullfill the `expression` condition on
    `table` in blocks of size `blen`.

    Parameters
    ----------
    expression : string or barray
        A boolean Numexpr expression or a boolean barray.
    blen : int
        The length of the block that is returned.  The default is the
        chunklen, or for a btable, the minimum of the different column
        chunklens.
    outfields : list of strings or string
        The list of column names that you want to get back in results.
        Alternatively, it can be specified as a string such as 'f0 f1' or
        'f0, f1'.
    limit : int
        A maximum number of elements to return.  The default is return
        everything.
    skip : int
        An initial number of elements to skip.  The default is 0.

    Returns
    -------
    out : iterable
        This iterable returns buffers as NumPy arrays made of
        structured types (or homogeneous ones in case `outfields` is a
        single field.

    See Also
    --------
    iterblocks

    """

    if blen is None:
        # Get the minimum chunklen for every field
        blen = min(table[col].chunklen for col in table.cols)
    if outfields is None:
        dtype = table.dtype
    else:
        if not isinstance(outfields, (list, tuple)):
            raise ValueError("only a sequence is supported for outfields")
        # Get the dtype for the outfields set
        try:
            dtype = [(name, table[name].dtype) for name in outfields]
        except IndexError:
            raise ValueError("Some names in `outfields` are not real fields")

    buf = np.empty(blen, dtype=dtype)
    nrow = 0
    for row in table.where(expression, outfields, limit, skip):
        buf[nrow] = row
        nrow += 1
        if nrow == blen:
            yield buf
            buf = np.empty(blen, dtype=dtype)
            nrow = 0
    yield buf[:nrow]


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
