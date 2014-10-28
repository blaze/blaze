from blaze.expr import (Expr, Symbol, Field, Arithmetic, RealMath, Map, Not,
        USub, Date, Time, DateTime, Millisecond, Microsecond, broadcast, sin,
        cos, isnan, UnaryOp)
import datetime
from datashape import iscollection
import math
import toolz
import itertools


funcnames = ('func_%d' % i for i in itertools.count())

def parenthesize(s):
    if ' ' in s:
        return '(%s)' % s
    else:
        return s


def print_numexpr(leaves, expr):
    """ Print expression to be evaluated in Python

    >>> from blaze.expr import ceil, sin

    >>> t = Symbol('t', 'var * {x: int, y: int, z: int, when: datetime}')
    >>> print_numexpr([t], t.x + t.y)
    'x + y'

    Supports mathematical functions

    >>> print_numexpr([t], sin(t.x) > cos(t.y))
    'sin(x) > cos(y)'

    Returns
    -------

    s: string
       A evalable string
    """
    if not isinstance(expr, Expr):
        return repr(expr)
    if any(expr.isidentical(leaf) for leaf in leaves):
        return expr._name
    if isinstance(expr, Symbol):
        return expr._name
    if isinstance(expr, Field):
        return expr._name
    if isinstance(expr, Arithmetic):
        lhs = print_numexpr(leaves, expr.lhs)
        rhs = print_numexpr(leaves, expr.rhs)
        return '%s %s %s' % (parenthesize(lhs),
                             expr.symbol,
                             parenthesize(rhs))
    if isinstance(expr, RealMath):
        child = print_numexpr(leaves, expr._child)
        return '%s(%s)' % (type(expr).__name__, child)
    if isinstance(expr, UnaryOp) and hasattr(expr, 'symbol'):
        child = print_numexpr(leaves, expr._child)
        return '%s%s' % (expr.symbol, parenthesize(child))
    if isinstance(expr, isnan):
        child = print_numexpr(leaves, expr._child)
        return '%s != %s' % (parenthsize(child), parenthesize(child))
    raise NotImplementedError()


WantToBroadcast = (Arithmetic, RealMath, Not, USub)
Broadcastable = (Arithmetic, RealMath, Not, USub)


def broadcast_numexpr_collect(expr, Broadcastable=Broadcastable,
                                    WantToBroadcast=WantToBroadcast):
    """ Collapse expression down using Broadcast - appropriate for numexpr

    Expressions of type Broadcastables are swallowed into Broadcast
    operations

    >>> x = Symbol('x', '5 * 3 * int')
    >>> y = Symbol('y', '5 * 3 * int')

    >>> expr = 2 * x + y
    >>> broadcast_numexpr_collect(expr)
    Broadcast(_children=(x, y), _scalars=(x, y), _scalar_expr=(2 * x) + y)

    >>> t = Symbol('t', 'var * {x: int, y: int, z: int, when: datetime}')
    >>> expr = (t.x + 2*t.y).distinct()

    >>> broadcast_numexpr_collect(expr)
    distinct(Broadcast(_children=(t.x, t.y), _scalars=(x, y), _scalar_expr=x + (2 * y)))
    """
    if (isinstance(expr, WantToBroadcast) and
        iscollection(expr.dshape)):
        leaves = leaves_of_type(Broadcastable, expr)
        expr = broadcast(expr, sorted(leaves, key=str))

    # Recurse down
    children = list(map(broadcast_numexpr_collect, expr._inputs))
    return expr._subs(dict(zip(expr._inputs, children)))


@toolz.curry
def leaves_of_type(types, expr):
    """ Leaves of an expression skipping all operations of type ``types``
    """
    if not isinstance(expr, types):
        return set([expr])
    else:
        return set.union(*map(leaves_of_type(types), expr._inputs))
