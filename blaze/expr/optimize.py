from .expressions import *
from .collections import *
from .split_apply_combine import *
from .broadcast import *
from .reductions import *
from ..dispatch import dispatch


def lean_projection(expr):
    """ Insert projections to keep dataset as thin as possible

    >>> t = Symbol('t', 'var * {a: int, b: int, c: int, d: int}')
    >>> expr = t[t.a > 0].b
    >>> lean_projection(expr)
    t[['a', 'b']][t.a > 0].b
    """
    fields = expr.fields

    return _lean(expr, fields=expr.fields)[0]


@dispatch(Symbol)
def _lean(expr, fields=None):
    return expr[sorted(fields)], fields


@dispatch(Projection)
def _lean(expr, fields=None):
    child, _ = _lean(expr._child, fields=fields)
    return child[sorted(fields, key=expr.fields.index)], fields


@dispatch(Field)
def _lean(expr, fields=None):
    fields = set(fields)
    fields.add(expr._name)
    child, _ = _lean(expr._child, fields=fields)
    return child[expr._name], fields


@dispatch(Broadcast)
def _lean(expr, fields=None):
    fields = set(fields) | set(expr.active_columns())
    child, _ = _lean(expr._child, fields=fields)
    return expr._subs({expr._child: child}), fields


@dispatch(Selection)
def _lean(expr, fields=None):
    predicate, pred_fields = _lean(expr.predicate, fields=fields)
    fields = set(fields) | set(pred_fields)
    child, _ = _lean(expr._child, fields=fields)
    return expr._subs({expr._child: child}), fields


@dispatch(Reduction)
def _lean(expr, fields=None):
    child, child_fields = _lean(expr._child, fields=set())
    return expr._subs({expr._child: child}), child_fields


@dispatch(Summary)
def _lean(expr, fields=None):
    values = []
    fields = set()
    for v in expr.values:
        child, child_fields = _lean(v, fields=set())
        values.append(child)
        fields |= set(child_fields)

    child, fields = _lean(expr._child, fields=fields)
    return expr._subs({expr._child: child}), fields


@dispatch(By)
def _lean(expr, fields=None):
    grouper, grouper_fields = _lean(expr.grouper, fields=set())
    apply, apply_fields = _lean(expr.apply, fields=set())

    fields = set(apply_fields) | set(grouper_fields)


    child, _ = _lean(expr._child, fields=fields)
    return expr._subs({expr._child: child}), fields
