from __future__ import absolute_import, division, print_function

from ..expr import (Expr, Symbol, Field, Arithmetic, UnaryMath, Not, USub,
                    isnan, UnaryOp, BinOp)
from toolz import curry
import itertools
from ..expr.broadcast import broadcast_collect


funcnames = ('func_%d' % i for i in itertools.count())


def parenthesize(s):
    if ' ' in s:
        return '(%s)' % s
    else:
        return s


def print_numexpr(leaves, expr):
    """ Print expression to be evaluated in Python

    >>> from blaze import symbol, ceil, sin, cos

    >>> t = symbol('t', 'var * {x: int, y: int, z: int, when: datetime}')
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
    if isinstance(expr, UnaryMath):
        child = print_numexpr(leaves, expr._child)
        return '%s(%s)' % (type(expr).__name__, child)
    if isinstance(expr, UnaryOp) and hasattr(expr, 'symbol'):
        child = print_numexpr(leaves, expr._child)
        return '%s%s' % (expr.symbol, parenthesize(child))
    if isinstance(expr, isnan):
        child = print_numexpr(leaves, expr._child)
        return '%s != %s' % (parenthesize(child), parenthesize(child))
    raise NotImplementedError("Operation %s not supported by numexpr" %
                              type(expr).__name__)


Broadcastable = WantToBroadcast = BinOp, UnaryOp

broadcast_numexpr_collect = curry(
    broadcast_collect,
    broadcastable=Broadcastable,
    want_to_broadcast=WantToBroadcast
)
