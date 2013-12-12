# -*- coding: utf-8 -*-

"""
Reduction functions.
"""

from __future__ import print_function, division, absolute_import

from .ufuncs import logical_and, logical_or, abs
from ..function import function, elementwise

#------------------------------------------------------------------------
# Reduce Impl
#------------------------------------------------------------------------

def reduce(kernel, a, axis=None):
    if axis is None:
        axes = range(a.ndim)
    elif isinstance(axis, int):
        axes = (axis,)
    else:
        axes = axis # Tuple

    # TODO: validate axes
    # TODO: insert map for other dimensions
    result = a
    for axis in axes:
        result = reduce_dim(kernel, result)
    return result

# TODO: Deferred
# @kernel('* -> X, Y..., Dtype -> Y..., Dtype')
def reduce_dim(kernel, a):
    from blaze import eval

    a = eval(a)
    it = iter(a)
    result = next(it)
    for x in it:
        result = kernel(result, x)

    return result

#------------------------------------------------------------------------
# Higher-level reductions
#------------------------------------------------------------------------

def all(a):
    return reduce(logical_and, a)

def any(a):
    return reduce(logical_or, a)

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
    return all(abs(a - b) <= atol + rtol * abs(b))
