from .core import pre_compute
from ..compatibility import collections_abc
from ..dispatch import dispatch
from ..expr import Expr
from odo.backends.json import JSON, JSONLines
from odo import into
from odo.utils import records_to_tuples


__all__ = ['pre_compute']


@dispatch(Expr, JSON)
def pre_compute(expr, data, **kwargs):
    seq = into(list, data, **kwargs)
    leaf = expr._leaves()[0]
    return list(records_to_tuples(leaf.dshape, seq))


@dispatch(Expr, JSONLines)
def pre_compute(expr, data, **kwargs):
    seq = into(collections_abc.Iterator, data, **kwargs)
    leaf = expr._leaves()[0]
    return records_to_tuples(leaf.dshape, seq)
