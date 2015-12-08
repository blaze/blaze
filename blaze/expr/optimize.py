from __future__ import absolute_import, division, print_function
from copy import deepcopy

from datashape.predicates import isscalar
from multipledispatch import MDNotImplementedError

from .expressions import *
from .strings import *
from .arithmetic import *
from .collections import *
from .split_apply_combine import *
from .broadcast import *
from .reductions import *
from ..dispatch import dispatch


def lean_projection(expr):
    """ Insert projections to keep dataset as thin as possible

    >>> t = symbol('t', 'var * {a: int, b: int, c: int, d: int}')
    >>> lean_projection(t.sort('a').b)
    t[['a', 'b']].sort('a', ascending=True).b
    """
    fields = expr.fields
    return _lean(expr, fields=fields)[0]


@dispatch(Symbol)
def _lean(expr, fields=None):
    """

    >>> s = symbol('s', '{x: int, y: int}')
    >>> _lean(s, ('x',))
    (s['x'], ('x',))

    >>> _lean(s, ())
    (s, ())

    >>> s = symbol('s', 'int')
    >>> _lean(s, ())
    (s, ())
    >>> _lean(s, ('s',))
    (s, ())
    """
    if not fields or set(expr.fields).issubset(fields):
        return expr, fields
    else:
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


@dispatch(Arithmetic)
def _lean(expr, fields=None):
    lhs, right_fields = _lean(expr.lhs, fields=())
    rhs, left_fields = _lean(expr.rhs, fields=())
    new_fields = set(fields) | set(left_fields) | set(right_fields)

    return type(expr)(lhs, rhs), new_fields


@dispatch(object)
def _lean(expr, fields=None):
    return expr, fields


@dispatch(Label)
def _lean(expr, fields=None):
    child, new_fields = _lean(expr._child, fields=())
    return child.label(expr._name), new_fields


@dispatch(ReLabel)
def _lean(expr, fields=None):
    labels = dict(expr.labels)
    reverse_labels = dict((v, k) for k, v in expr.labels)

    child_fields = set(reverse_labels.get(f, f) for f in fields)

    child, new_fields = _lean(expr._child, fields=child_fields)
    return child.relabel(**dict((k, v) for k, v in expr.labels if k in
        child.fields)), new_fields


@dispatch(ElemWise)
def _lean(expr, fields=None):
    if isscalar(expr._child.dshape.measure):
        child, _ = _lean(expr._child, fields=set(expr._child.fields))
        return expr._subs({expr._child: child}), set(expr._child.fields)
    else:
        raise MDNotImplementedError()


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


@dispatch(Sort)
def _lean(expr, fields=None):
    key = expr.key
    if not isinstance(key, (list, set, tuple)):
        key = [key]
    new_fields = set(fields) | set(key)

    child, _ = _lean(expr._child, fields=new_fields)
    return child.sort(key=expr.key, ascending=expr.ascending), new_fields


@dispatch(Head)
def _lean(expr, fields=None):
    child, child_fields = _lean(expr._child, fields=fields)
    return child.head(expr.n), child_fields


@dispatch(Reduction)
def _lean(expr, fields=None):
    child = expr._child
    try:
        fields = child.active_columns()
    except AttributeError:
        fields = child.fields
    child, child_fields = _lean(child, fields=set(filter(None, fields)))
    return expr._subs({expr._child: child}), child_fields


@dispatch(Summary)
def _lean(expr, fields=None):
    save = dict()
    new_fields = set()
    for name, val in zip(expr.names, expr.values):
        if name not in fields:
            continue
        child, child_fields = _lean(val, fields=set())
        save[name] = child
        new_fields |= set(child_fields)

    return summary(**save), new_fields


@dispatch(By)
def _lean(expr, fields=None):
    fields = set(fields)
    grouper, grouper_fields = _lean(expr.grouper,
                                    fields=fields.intersection(expr.grouper.fields))
    apply, apply_fields = _lean(expr.apply,
                                fields=fields.intersection(expr.apply.fields))

    new_fields = set(apply_fields) | set(grouper_fields)

    child = common_subexpression(grouper, apply)
    if len(child.fields) > len(new_fields):
        child, _ = _lean(child, fields=new_fields)
        grouper = grouper._subs({expr._child: child})
        apply = apply._subs({expr._child: child})

    return By(grouper, apply), new_fields


@dispatch(Distinct)
def _lean(expr, fields=None):
    child, new_fields = _lean(expr._child, fields=expr.fields)
    return expr._subs({expr._child: child}), new_fields


@dispatch(Merge)
def _lean(expr, fields=None):
    new_fields = set()
    for f in expr.fields:
        if f not in fields:
            continue
        le, nf = _lean(expr[f], fields=set([f]))
        new_fields.update(nf)
    child, _ = _lean(expr._child, fields=new_fields)

    return expr._subs({expr._child: child})[sorted(fields)], new_fields


@dispatch((Join, Concat))
def _lean(expr, fields=None):
    return expr, fields


@dispatch(Expr)
def _lean(expr, fields=None):
    """ Lean projection version of expression

    Parameters
    ----------

    expr : Expression
        An expression to be optimized
    fields : Iterable of strings
        The fields that will be needed from this expression

    Returns
    -------

    expr : Expression
        An expression with Projections inserted to avoid unnecessary fields
    fields : Iterable of strings
        The fields that this expression requires to execute
    """
    raise NotImplementedError()


@dispatch(Selection)
def simple_selections(expr):
    """Cast all ``Selection`` nodes into ``SimpleSelection`` nodes.

    This causes the compute core to not treat the predicate as an input.

    Parameters
    ----------
    expr : Expr
        The expression to traverse.

    Returns
    -------
    siplified : Expr
        The expression with ``Selection``s replaces with ``SimpleSelection``s.
    """
    return SimpleSelection(
        simple_selections(expr._child),
        simple_selections(expr.predicate),
    )


@dispatch(Expr)
def simple_selections(expr):
    return expr._subs({e: simple_selections(e) for e in expr._inputs})
