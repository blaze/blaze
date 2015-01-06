from __future__ import absolute_import, division, print_function

import numpy as np

from .core import compute
from ..expr import Broadcast, symbol, UTCFromTimestamp, DateTimeTruncate
from ..expr import Map
from ..dispatch import dispatch
from toolz import memoize
import datashape
import numba
from .pyfunc import funcstr


def get_numba_type(dshape):
    """Get the ``numba`` type corresponding to the ``datashape.Mono`` instance
    `dshape`

    Parameters
    ----------
    dshape : datashape.Mono

    Returns
    -------
    restype : numba.types.Type

    Examples
    --------
    >>> import datashape
    >>> import numba

    >>> get_numba_type(datashape.bool_)
    bool

    >>> get_numba_type(datashape.date_)
    datetime64(D)

    >>> get_numba_type(datashape.datetime_)
    datetime64(us)

    >>> get_numba_type(datashape.timedelta_)  # default unit is microseconds
    timedelta64(us)

    >>> get_numba_type(datashape.TimeDelta('D'))
    timedelta64(D)

    >>> get_numba_type(datashape.int64)
    int64

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
    else:
        restype = getattr(numba, str(measure))
    return restype


def compute_signature(expr):
    """Get the ``numba`` *function signature* corresponding to ``DataShape``

    Examples
    --------
    >>> from blaze import symbol
    >>> s = symbol('s', 'int64')
    >>> t = symbol('t', 'float32')
    >>> d = symbol('d', 'datetime')

    >>> expr = s + t
    >>> compute_signature(expr)
    float64(int64, float32)

    >>> expr = d.truncate(days=1)
    >>> compute_signature(expr)
    datetime64(D)(datetime64(us))

    >>> expr = d.day + 1
    >>> compute_signature(expr)  # only looks at leaf nodes
    int64(datetime64(us))

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
    leaves = expr._leaves()
    s, scope = funcstr(leaves, expr)
    scope = {k: numba.jit(nopython=True)(v) if callable(v) else v
             for k, v in scope.items()}
    func = eval(s, scope)
    sig = compute_signature(expr)
    return numba.vectorize([sig], nopython=True)(func)


# do this here so we can run our doctest
get_numba_ufunc = memoize(_get_numba_ufunc)


@dispatch(Broadcast, np.ndarray)
def compute_up(t, x, **kwargs):
    assert len(t._scalars) == 1
    scalar = t._scalars[0]
    fields = scalar.fields
    d = dict((scalar[c], symbol(c, getattr(scalar, c).dshape))
             for i, c in enumerate(fields))
    expr = t._scalar_expr._subs(d)

    if isinstance(expr, (UTCFromTimestamp, DateTimeTruncate)):
        # numba segfaults here
        return compute(t._scalar_expr, x)
    else:
        ufunc = get_numba_ufunc(expr)
        if x.dtype.names is not None:
            return ufunc(*(x[leaf._name] for leaf in expr._leaves()))
        else:
            return ufunc(x)
