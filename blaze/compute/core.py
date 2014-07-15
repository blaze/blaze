from __future__ import absolute_import, division, print_function
from datetime import date, datetime
from toolz import first

from ..expr.core import *
from ..expr.table import *
from ..expr.scalar import *
from ..dispatch import dispatch

__all__ = ['compute', 'compute_one']

base = (int, float, str, bool, date, datetime)


@dispatch(base)
def compute_one(a, **kwargs):
    return a


@dispatch(Expr, object)
def compute(expr, o, **kwargs):
    """ Compute against single input

    Assumes that only one TableSymbol exists in expression

    >>> t = TableSymbol('t', '{name: string, balance: int}')
    >>> deadbeats = t[t['balance'] < 0]['name']

    >>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
    >>> # list(compute(deadbeats, {t: data}))
    >>> list(compute(deadbeats, data))
    ['Bob', 'Charlie']
    """
    ts = set([x for x in expr.ancestors() if isinstance(x, TableSymbol)])
    if len(ts) == 1:
        return compute(expr, {first(ts): o}, **kwargs)
    else:
        raise ValueError("Give compute dictionary input, got %s" % str(o))


def bottom_up(d, expr):
    """
    Process an expression from the leaves upwards

    Parameters
    ----------

    d : dict mapping {TableSymbol: data}
        Maps expressions to data elements, likely at the leaves of the tree
    expr : Expr
        Expression to compute

    Helper function for ``compute``
    """
    # Base case: expression is in dict, return associated data
    if expr in d:
        return d[expr]

    # Compute children of this expression
    children = ([bottom_up(d, child) for child in expr.inputs]
                if hasattr(expr, 'inputs') else [])

    # Compute this expression given the children
    result = compute_one(expr, *children, scope=d)

    return result


@dispatch(Expr, dict)
def pre_compute(expr, d):
    """ Transform expr prior to calling ``compute`` """
    return expr


@dispatch(Expr, object, dict)
def post_compute(expr, result, d):
    """ Effects after the computation is complete """
    return result


@dispatch(Expr, dict)
def compute(expr, d):
    """ Compute expression against data sources

    >>> t = TableSymbol('t', '{name: string, balance: int}')
    >>> deadbeats = t[t['balance'] < 0]['name']

    >>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
    >>> list(compute(deadbeats, {t: data}))
    ['Bob', 'Charlie']
    """
    expr = pre_compute(expr, d)
    result = bottom_up(d, expr)
    return post_compute(expr, result, d)


def columnwise_funcstr(t, variadic=True, full=False):
    """
    >>> t = TableSymbol('t', '{x: real, y: real, z: real}')
    >>> cw = t['x'] + t['z']
    >>> columnwise_funcstr(cw)
    'lambda x, z: x + z'

    >>> columnwise_funcstr(cw, variadic=False)
    'lambda (x, z): x + z'

    >>> columnwise_funcstr(cw, variadic=False, full=True)
    'lambda (x, y, z): x + z'
    """
    if full:
        columns = t.child.columns
    else:
        columns = t.active_columns()
    if variadic:
        prefix = 'lambda %s: '
    else:
        prefix = 'lambda (%s): '

    return prefix % ', '.join(map(str, columns)) + eval_str(t.expr)
