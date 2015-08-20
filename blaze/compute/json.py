from .core import pre_compute
from ..dispatch import dispatch
from ..expr import Expr
from odo.backends.json import JSON, JSONLines
from odo import into
from collections import Iterator
from odo.utils import records_to_tuples


__all__ = ['pre_compute']


@dispatch(Expr, JSON)
def pre_compute(expr, data, **kwargs):
    seq = into(list, data, **kwargs)
    leaf = expr._leaves()[0]
    return list(records_to_tuples(leaf.dshape, seq))


@dispatch(Expr, JSONLines)
def pre_compute(expr, data, **kwargs):
    seq = into(Iterator, data, **kwargs)
    leaf = expr._leaves()[0]
    return records_to_tuples(leaf.dshape, seq)
