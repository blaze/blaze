from __future__ import absolute_import, division, print_function

from datetime import datetime, date

from ..expr.table import *
from ..expr.scalar import *
from ..dispatch import dispatch

__all__ = ['compute']

base = (int, float, str, bool, date, datetime)


@dispatch(base, object)
def compute(a, b):
    return a


@dispatch(TableExpr, dict)
def compute(t, d):
    if t in d:
        return d[t]

    # Pare down d to only nodes in table
    nodes = set(t.traverse())
    d = dict((k, v) for k, v in d.items()
                    if k in nodes)

    # Common case: One relevant value in dict
    #              Switch to standard dispatching scheme
    if len(d) == 1:
        return compute(t, list(d.values())[0])

    if hasattr(t, 'parent'):
        parent = compute(t.parent, d)
        t2 = t.subs({t.parent: TableSymbol('', t.parent.schema)})
        return compute(t2, parent)

    raise NotImplementedError("No method found to compute on multiple Tables")


@dispatch(Join, dict)
def compute(t, d):
    lhs = compute(t.lhs, d)
    rhs = compute(t.rhs, d)


    t2 = t.subs({t.lhs: TableSymbol('_lhs', t.lhs.schema),
                 t.rhs: TableSymbol('_rhs', t.rhs.schema)})
    return compute(t2, lhs, rhs)


@dispatch(Join, object)
def compute(t, o):
    return compute(t, o, o)


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
