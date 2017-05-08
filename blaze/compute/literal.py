from odo import append, drop

from ..compute.core import compute
from ..dispatch import dispatch
from ..expr.expressions import Expr
from ..expr.literal import BoundSymbol


@dispatch(Expr, BoundSymbol)
def compute_down(expr, data, **kwargs):
    return compute(expr, data.data, **kwargs)


@dispatch(Expr, BoundSymbol)
def pre_compute(expr, data, **kwargs):
    return pre_compute(expr, data.data, **kwargs)


@dispatch(BoundSymbol, object)
def append(a, b, **kwargs):
    return append(a.data, b, **kwargs)


@dispatch(BoundSymbol)
def drop(d):
    return drop(d.data)
