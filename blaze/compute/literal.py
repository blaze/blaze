from odo import append, drop

from ..compute.core import compute
from ..dispatch import dispatch
from ..expr.expressions import Expr
from ..expr.literal import Data


@dispatch(Expr, Data)
def compute_down(expr, dta, **kwargs):
    return compute(expr, dta.data, **kwargs)


@dispatch(Expr, Data)
def pre_compute(expr, dta, **kwargs):
    return pre_compute(expr, dta.data, **kwargs)


@dispatch(Data, object)
def append(a, b, **kwargs):
    return append(a.data, b, **kwargs)


@dispatch(Data)
def drop(d):
    return drop(d.data)
