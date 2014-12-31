from __future__ import absolute_import, division, print_function

import numpy as np

from .core import compute
from ..expr import Broadcast, symbol, UTCFromTimestamp, DateTimeTruncate
from ..dispatch import dispatch
from toolz import memoize
import datashape
import numba
from .pyfunc import lambdify


def get_numba_type(dshape):
    measure = dshape.measure
    if measure == datashape.bool_:
        restype = numba.bool_  # str(bool_) == 'bool' so we can't use getattr
    elif measure == datashape.date_:
        restype = numba.types.NPDatetime('D')
    elif measure == datashape.datetime_:
        restype = numba.types.NPDatetime('us')
    elif measure == datashape.timedelta_:
        restype = numba.types.NPTimedelta(measure.unit)
    else:
        restype = getattr(numba, str(measure))
    return restype


def compute_signature(expr):
    restype = get_numba_type(expr.schema)
    argtypes = [get_numba_type(e.schema) for e in expr._leaves()]
    return restype(*argtypes)


@memoize
def get_numba_ufunc(expr):
    leaves = expr._leaves()
    func = lambdify(leaves, expr)
    sig = compute_signature(expr)

    # we need getattr(..., 'func', func) for Map expressions which can be passed
    # directly into numba
    return numba.vectorize([sig], nopython=True)(getattr(expr, 'func', func))


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
