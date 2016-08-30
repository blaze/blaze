from .expressions import *
from .expressions import common_subexpression
from .collections import Merge, merge
from .reductions import Summary, summary
from .broadcast import Broadcast
from .split_apply_combine import By
from ..dispatch import dispatch


def rewrite(expr):
    return _rewrite(expr)


@dispatch(object)
def _rewrite(expr):
    return expr


@dispatch(Expr)
def _rewrite(expr):
    args = [_rewrite(a) for a in expr._args]
    return type(expr)(*args)


@dispatch(Projection)
def _rewrite(expr):
    child = _rewrite(expr._child)
    return child._project(expr._fields)


@dispatch((Selection, SimpleSelection))
def _rewrite(expr):
    _child, predicate = expr._args
    table = _rewrite(_child)
    predicate = _rewrite(predicate)
    subexpr = common_subexpression(table, predicate)
    return table._subs({subexpr: type(expr)(subexpr, predicate)})


@dispatch(Summary)
def _rewrite(expr):
    _, names, values, axis, keepdims = expr._args
    kwargs = dict()
    for name, val in zip(names, values):
        child = _rewrite(val)
        kwargs[name] = child

    return summary(keepdims=keepdims, axis=axis, **kwargs)


@dispatch(Field)
def _rewrite(expr):
    child = _rewrite(expr._child)
    if isinstance(child, Merge):
        return child._get_field(expr._name)
    return Field(child, expr._name)



