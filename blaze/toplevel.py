import os, os.path

from urlparse import urlparse
from params import params, to_bparams
from params import params as _params
from sources.sql import SqliteSource
from sources.chunked import BArraySource, BTableSource

from table import NDArray, Array, NDTable, Table
from blaze.datashape import from_numpy, to_numpy, TypeVar, Fixed
from blaze.expr import graph, ops
from blaze import blz, dshape as _dshape
from eclass import eclass as _eclass

import numpy as np

# TODO: we'd like to distinguish between opening in Deferred or
# Immediete mode

def open(uri, mode='a',  eclass=_eclass.manifest):
    """Open a Blaze object via an `uri` (Uniform Resource Identifier).

    Parameters
    ----------
    uri : str
        Specifies the URI for the Blaze object.  It can be a regular file too.
        The URL scheme indicates the storage type:

          * barray: BLZ array
          * btable: BLZ table
          * sqlite: SQLite table (the URI 'sqlite://' creates in-memory table)

        If no URI scheme is given, carray is assumed.

    mode : the open mode (string)
        Specifies the mode in which the object is opened.  The supported
        values are:

          * 'r' for read-only
          * 'w' for emptying the previous underlying data
          * 'a' for allowing read/write on top of existing data

    Returns
    -------
    out : an Array or Table object.

    """
    ARRAY = 1
    TABLE = 2

    uri = urlparse(uri)
    path = uri.netloc + uri.path
    parms = params(storage=path)

    if uri.scheme == 'carray':
        source = BArraySource(params=parms)
        structure = ARRAY

    elif uri.scheme == 'ctable':
        source = BTableSource(params=parms)
        structure = TABLE

    elif uri.scheme == 'sqlite':
        # Empty path means memory storage
        parms = params(storage=path or None)
        source = SqliteSource(params=parms)
        structure = TABLE

    else:
        # Default is to treat the URI as a regular path
        parms = params(storage=path)
        source = BArraySource(params=parms)
        structure = ARRAY

    # Don't want a deferred array (yet)
    # return NDArray(source)
    if structure == ARRAY:

        if eclass is _eclass.manifest:
            return Array(source)
        elif eclass is _eclass.delayed:
            return NDArray(source)

    elif structure == TABLE:

        if eclass is _eclass.manifest:
            return Table(source)
        elif eclass is _eclass.delayed:
            return NDTable(source)

# These are like NumPy equivalent except that they can allocate
# larger than memory.

def zeros(dshape, params=None, eclass=_eclass.manifest):
    """ Create an Array and fill it with zeros.

    Parameters
    ----------
    dshape : str, blaze.dshape instance
        Specifies the datashape of the outcome object.
    params : blaze.params object
        Any parameter supported by the backend library.

    Returns
    -------
    out : an Array object.

    """
    if isinstance(dshape, basestring):
        dshape = _dshape(dshape)
    shape, dtype = to_numpy(dshape)
    bparams, rootdir, format_flavor = to_bparams(params or _params())
    if rootdir is not None:
        blz.zeros(shape, dtype, rootdir=rootdir, bparams=bparams)
        return open(rootdir)
    else:
        source = BArraySource(blz.zeros(shape, dtype, bparams=bparams),
                              params=params)
        if eclass is _eclass.manifest:
            return Array(source)
        elif eclass is _eclass.delayed:
            return NDArray(source)

def ones(dshape, params=None, eclass=_eclass.manifest):
    """ Create an Array and fill it with ones.

    Parameters
    ----------
    dshape : str, blaze.dshape instance
        Specifies the datashape of the outcome object.
    params : blaze.params object
        Any parameter supported by the backend library.

    Returns
    -------
    out : an Array object.

    """
    if isinstance(dshape, basestring):
        dshape = _dshape(dshape)
    shape, dtype = to_numpy(dshape)
    bparams, rootdir, format_flavor = to_bparams(params or _params())
    if rootdir is not None:
        blz.ones(shape, dtype, rootdir=rootdir, bparams=bparams)
        return open(rootdir)
    else:
        source = BArraySource(blz.ones(shape, dtype, bparams=bparams),
                              params=params)
        if eclass is _eclass.manifest:
            return Array(source)
        elif eclass is _eclass.delayed:
            return NDArray(source)

def fromiter(iterable, dshape, params=None):
    """ Create an Array and fill it with values from `iterable`.

    Parameters
    ----------
    iterable : iterable object
        An iterable object providing data for the blz.
    dshape : str, blaze.dshape instance
        Specifies the datashape of the outcome object.  Only 1d shapes
        are supported right now. When the `iterator` should return an
        unknown number of items, a ``TypeVar`` can be used.
    params : blaze.params object
        Any parameter supported by the backend library.

    Returns
    -------
    out : an Array object.

    """
    if isinstance(dshape, basestring):
        dshape = _dshape(dshape)
    shape, dtype = dshape.parameters[:-1], dshape.parameters[-1]
    # Check the shape part
    if len(shape) > 1:
        raise ValueError("shape can be only 1-dimensional")
    length = shape[0]
    count = -1
    if type(length) == TypeVar:
        count = -1
    elif type(length) == Fixed:
        count = length.val

    dtype = dtype.to_dtype()
    # Now, create the Array itself (using the carray backend)
    bparams, rootdir, format_flavor = to_bparams(params or _params())
    if rootdir is not None:
        blz.fromiter(iterable, dtype, count=count,
                        rootdir=rootdir, bparams=bparams)
        return open(rootdir)
    else:
        ica = blz.fromiter(iterable, dtype, count=count, bparams=bparams)
        source = BArraySource(ica, params=params)
        return Array(source)

def loadtxt(filetxt, storage):
    """ Convert txt file into Blaze native format """
    Array(np.loadtxt(filetxt), params=params(storage=storage))

def lazy(a):
    """
    Turn an object into its lazy blaze equivalent.
    """
    # TODO: tables, etc
    if not isinstance(a, (NDArray, Array, graph.ExpressionNode)):
        a = NDArray(a)
    return a

def allclose(a, b, rtol=1e-05, atol=1e-08):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    If either array contains one or more NaNs, False is returned.
    Infs are treated as equal if they are in the same place and of the same
    sign in both arrays.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).

    Returns
    -------
    allclose : bool
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise.

    See Also
    --------
    all, any, alltrue, sometrue

    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    The above equation is not symmetric in `a` and `b`, so that
    `allclose(a, b)` might be different from `allclose(b, a)` in
    some rare cases.

    Examples
    --------
    >>> blaze.allclose([1e10,1e-7], [1.00001e10,1e-8])
    False
    >>> blaze.allclose([1e10,1e-8], [1.00001e10,1e-9])
    True
    >>> blaze.allclose([1e10,1e-8], [1.0001e10,1e-9])
    False
    >>> blaze.allclose([1.0, np.nan], [1.0, np.nan])
    False
    """
    a, b = lazy(a), lazy(b)
    return blaze_all(blaze_abs(a - b) <= atol + rtol * blaze_abs(b))

def blaze_all(a, axis=None, out=None):
    """
    Test whether all array elements along a given axis evaluate to True.
    """
    a = lazy(a)
    return a.all(axis=axis, out=out)

def blaze_any(a, axis=None, out=None):
    """
    Test whether any array elements along a given axis evaluate to True.
    """
    a = lazy(a)
    return a.any(axis=axis, out=out)

def blaze_abs(a, axis=None, out=None):
    """
    Returns the absolute value element-wise.
    """
    a = lazy(a)
    op = ops.Abs('Abs', [a], {'out': out})
    return op

def blaze_sum(a, axis=None, out=None):
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Elements to sum.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.
        The default (`axis` = `None`) is perform a sum over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.

        .. versionadded:: 1.7.0

        If this is a tuple of ints, a sum is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : ndarray, optional
        Array into which the output is placed.  By default, a new array is
        created.  If `out` is given, it must be of the appropriate shape
        (the shape of `a` with `axis` removed, i.e.,
        ``numpy.delete(a.shape, axis)``).  Its type is preserved. See
        `doc.ufuncs` (Section "Output arguments") for more details.

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.

    See Also
    --------
    ndarray.sum : Equivalent method.

    cumsum : Cumulative sum of array elements.

    trapz : Integration of array values using the composite trapezoidal rule.

    mean, average

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    Examples
    --------
    >>> np.sum([0.5, 1.5])
    2.0
    >>> np.sum([[0, 1], [0, 5]])
    6
    >>> np.sum([[0, 1], [0, 5]], axis=0)
    array([0, 6])
    >>> np.sum([[0, 1], [0, 5]], axis=1)
    array([1, 5])
    """
    a = lazy(a)
    return a.sum(axis=axis, out=out)
