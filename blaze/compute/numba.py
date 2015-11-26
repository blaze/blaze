from __future__ import absolute_import, division, print_function
import threading

import numpy as np
from toolz import memoize
import datashape
import numba

from datashape import isdatelike, TimeDelta

from .core import optimize
from ..expr.strings import isstring
from ..expr import Expr, Map, BinOp, UnaryOp
from ..expr.broadcast import broadcast_collect, Broadcast
from .pyfunc import funcstr


Broadcastable = BinOp, Map, UnaryOp
lock = threading.Lock()


def optimize_ndarray(expr, *data, **kwargs):
    for leaf in expr._leaves():
        leaf_measure = leaf.dshape.measure

        # TODO: remove datelike skipping when numba/numba#1202 is fixed
        if (isstring(leaf_measure) or
            isdatelike(leaf_measure) or
            isinstance(leaf_measure, TimeDelta) or
            isinstance(leaf.dshape.measure, datashape.Record) and
            any(isstring(dt) or isdatelike(dt) or isinstance(dt, TimeDelta)
                for dt in leaf.dshape.measure.types)):
            return expr
    else:
        return broadcast_collect(
            expr,
            broadcastable=Broadcastable,
            want_to_broadcast=Broadcastable,
        )


for i in range(1, 11):
    optimize.register(Expr, *([np.ndarray] * i))(optimize_ndarray)


def get_numba_type(dshape):
    """Get the numba type corresponding to the ``datashape.Mono`` instance.
    `dshape`

    Parameters
    ----------
    dshape : datashape.Mono

    Returns
    -------
    restype : numba.types.Type
        Numba type corresponding to `dshape`.

    Examples
    --------
    >>> get_numba_type(datashape.bool_) == numba.bool_
    True

    See Also
    --------
    compute_signature
    """
    measure = dshape.measure
    if measure == datashape.bool_:
        restype = numba.bool_  # str(bool_) == 'bool' so we can't use getattr
    elif measure == datashape.date_:
        restype = numba.types.NPDatetime('D')
    elif measure == datashape.datetime_:
        restype = numba.types.NPDatetime('us')
    elif isinstance(measure, datashape.TimeDelta):  # isinstance for diff freqs
        restype = numba.types.NPTimedelta(measure.unit)
    elif isinstance(measure, datashape.String):
        encoding = measure.encoding
        fixlen = measure.fixlen
        if fixlen is None:
            if encoding == 'A':
                return numba.types.string
            raise TypeError("Numba cannot handle variable length strings")
        typ = (numba.types.CharSeq
               if encoding == 'A' else numba.types.UnicodeCharSeq)
        return typ(fixlen or 0)
    elif measure == datashape.object_:
        raise TypeError("Numba cannot handle object datashape")
    else:
        try:
            restype = getattr(numba, str(measure))
        except AttributeError:
            raise TypeError('Invalid datashape to numba type: %r' % measure)
    return restype


def compute_signature(expr):
    """Get the ``numba`` *function signature* corresponding to ``DataShape``

    Examples
    --------
    >>> from blaze import symbol
    >>> s = symbol('s', 'int64')
    >>> t = symbol('t', 'float32')
    >>> from numba import float64, int64, float32
    >>> expr = s + t
    >>> compute_signature(expr) == float64(int64, float32)
    True

    Notes
    -----
    * This could potentially be adapted/refactored to deal with
      ``datashape.Function`` types.
    * Cannot handle ``datashape.Record`` types.
    """
    assert datashape.isscalar(expr.schema)
    restype = get_numba_type(expr.schema)
    argtypes = [get_numba_type(e.schema) for e in expr._leaves()]
    return restype(*argtypes)


def _get_numba_ufunc(expr):
    """Construct a numba ufunc from a blaze expression

    Parameters
    ----------
    expr : blaze.expr.Expr

    Returns
    -------
    f : function
        A numba vectorized function

    Examples
    --------
    >>> from blaze import symbol
    >>> import numpy as np

    >>> s = symbol('s', 'float64')
    >>> t = symbol('t', 'float64')

    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = np.array([2.0, 3.0, 4.0])

    >>> f = get_numba_ufunc(s + t)

    >>> f(x, y)
    array([ 3.,  5.,  7.])

    See Also
    --------
    get_numba_type
    compute_signature
    """
    if isinstance(expr, Broadcast):
        leaves = expr._scalars
        expr = expr._scalar_expr
    else:
        leaves = expr._leaves()

    s, scope = funcstr(leaves, expr)

    scope = dict((k, numba.jit(nopython=True)(v) if callable(v) else v)
                 for k, v in scope.items())
    # get the func
    func = eval(s, scope)
    # get the signature
    sig = compute_signature(expr)
    # vectorize is currently not thread safe. So lock the thread.
    # TODO FIXME remove this when numba has made vectorize thread safe.
    with lock:
        ufunc = numba.vectorize([sig], nopython=True)(func)
    return ufunc


# do this here so we can run our doctest
get_numba_ufunc = memoize(_get_numba_ufunc)


def broadcast_numba(t, *data, **kwargs):
    return get_numba_ufunc(t)(*data)
