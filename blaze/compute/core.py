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
def compute_one(a):
    return a

@dispatch(Expr, object)
def compute(expr, o):
    ts = set([x for x in expr.ancestors() if isinstance(x, TableSymbol)])
    if len(ts) == 1:
        return compute(expr, {first(ts): o})
    else:
        raise ValueError("Give compute dictionary input, got %s" % str(o))


def bottom_up(d, expr):
    try:
        return d[expr]
    except:
        pass

    if isinstance(expr, base):
        return expr

    parents = [bottom_up(d, getattr(expr, parent)) for parent in expr.inputs]

    result = compute_one(expr, *parents)

    return result

@dispatch(Expr, dict)
def compute(expr, d):
    return bottom_up(d, expr)


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
        columns = t.parent.columns
    else:
        columns = t.active_columns()
    if variadic:
        prefix = 'lambda %s: '
    else:
        prefix = 'lambda (%s): '

    return prefix % ', '.join(map(str, columns)) + eval_str(t.expr)
