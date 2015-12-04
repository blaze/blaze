from __future__ import absolute_import, division, print_function

import pandas as pd
from ..expr import (Expr, Symbol, Field, Arithmetic, UnaryMath, BinaryMath,
                    Date, Time, DateTime, Millisecond, Microsecond, broadcast,
                    sin, cos, Map, UTCFromTimestamp, DateTimeTruncate, symbol,
                    USub, Not, notnull, greatest, least, atan2, Like)
from ..expr import math as expr_math
from ..expr.expressions import valid_identifier
from ..dispatch import dispatch
from . import pydatetime
import numpy as np
import datetime
import fnmatch
import math
import toolz
import itertools


funcnames = ('func_%d' % i for i in itertools.count())


def parenthesize(s):
    if ' ' in s:
        return '(%s)' % s
    else:
        return s


def print_python(leaves, expr):
    """ Print expression to be evaluated in Python

    >>> from blaze.expr import ceil, sin

    >>> t = symbol('t', '{x: int, y: int, z: int, when: datetime}')
    >>> print_python([t], t.x + t.y)
    ('t[0] + t[1]', {})

    Supports mathematical and datetime access

    >>> print_python([t], sin(t.x) > ceil(t.y))  # doctest: +SKIP
    ('math.sin(t[0]) > math.ceil(t[1])', {'math':<module 'math'>})
    >>> print_python([t], t.when.day + 1)
    ('t[3].day + 1', {})

    Specify leaves of the expression to control level of printing

    >>> print_python([t.x, t.y], t.x + t.y)
    ('x + y', {})

    Returns
    -------

    s: string
       A evalable string
    scope: dict
       A namespace to add to be given to eval
    """
    if isinstance(expr, Expr) and any(expr.isidentical(lf) for lf in leaves):
        return valid_identifier(expr._name), {}
    return _print_python(expr, leaves=leaves)


@dispatch(object)
def _print_python(expr, leaves=None):
    return repr(expr), {}


@dispatch((datetime.datetime, datetime.date))
def _print_python(expr, leaves=None):
    return repr(expr), {'datetime': datetime, 'Timestamp': pd.Timestamp}


@dispatch(Symbol)
def _print_python(expr, leaves=None):
    return valid_identifier(expr._name), {}


@dispatch(Field)
def _print_python(expr, leaves=None):
    child, scope = print_python(leaves, expr._child)
    index = expr._child.fields.index(expr._name)
    return '%s[%d]' % (parenthesize(child), index), scope


@dispatch(Arithmetic)
def _print_python(expr, leaves=None):
    lhs, left_scope = print_python(leaves, expr.lhs)
    rhs, right_scope = print_python(leaves, expr.rhs)
    return ('%s %s %s' % (parenthesize(lhs),
                          expr.symbol,
                          parenthesize(rhs)),
            toolz.merge(left_scope, right_scope))


@dispatch(USub)
def _print_python(expr, leaves=None):
    child, scope = print_python(leaves, expr._child)
    return '%s%s' % (expr.symbol, parenthesize(child)), scope


@dispatch(Not)
def _print_python(expr, leaves=None):
    child, scope = print_python(leaves, expr._child)
    return 'not %s' % parenthesize(child), scope


@dispatch(UnaryMath)
def _print_python(expr, leaves=None):
    child, scope = print_python(leaves, expr._child)
    return ('np.%s(%s)' % (type(expr).__name__, child),
            toolz.merge(scope, {'np': np}))


@dispatch(BinaryMath)
def _print_python(expr, leaves=None):
    lhs, scope_lhs = print_python(leaves, expr.lhs)
    rhs, scope_rhs = print_python(leaves, expr.rhs)
    return ('np.%s(%s, %s)' % (type(expr).__name__, lhs, rhs),
            toolz.merge(scope_lhs, scope_rhs, {'np': np}))


@dispatch(atan2)
def _print_python(expr, leaves=None):
    lhs, scope_lhs = print_python(leaves, expr.lhs)
    rhs, scope_rhs = print_python(leaves, expr.rhs)
    return ('np.arctan2(%s, %s)' % (lhs, rhs),
            toolz.merge(scope_lhs, scope_rhs, {'np': np}))


@dispatch(greatest)
def _print_python(expr, leaves=None):
    lhs, scope_lhs = print_python(leaves, expr.lhs)
    rhs, scope_rhs = print_python(leaves, expr.rhs)
    return 'max(%s, %s)' % (lhs, rhs), toolz.merge(scope_lhs, scope_rhs)


@dispatch(least)
def _print_python(expr, leaves=None):
    lhs, scope_lhs = print_python(leaves, expr.lhs)
    rhs, scope_rhs = print_python(leaves, expr.rhs)
    return 'min(%s, %s)' % (lhs, rhs), toolz.merge(scope_lhs, scope_rhs)


@dispatch(expr_math.abs)
def _print_python(expr, leaves=None):
    child, scope = print_python(leaves, expr._child)
    return ('abs(%s)' % child, scope)


@dispatch(Date)
def _print_python(expr, leaves=None):
    child, scope = print_python(leaves, expr._child)
    return ('%s.date()' % parenthesize(child), scope)


@dispatch(Time)
def _print_python(expr, leaves=None):
    child, scope = print_python(leaves, expr._child)
    return ('%s.time()' % parenthesize(child), scope)


@dispatch(Millisecond)
def _print_python(expr, leaves=None):
    child, scope = print_python(leaves, expr._child)
    return ('%s.microsecond // 1000' % parenthesize(child), scope)


@dispatch(UTCFromTimestamp)
def _print_python(expr, leaves=None):
    child, scope = print_python(leaves, expr._child)
    return ('datetime.datetime.utcfromtimestamp(%s)' % parenthesize(child),
            toolz.merge({'datetime': datetime}, scope))


@dispatch(DateTime)
def _print_python(expr, leaves=None):
    child, scope = print_python(leaves, expr._child)
    attr = type(expr).__name__.lower()
    return ('%s.%s' % (parenthesize(child), attr), scope)


@dispatch(DateTimeTruncate)
def _print_python(expr, leaves=None):
    child, scope = print_python(leaves, expr._child)
    scope['truncate'] = pydatetime.truncate
    return ('truncate(%s, %s, "%s")' % (child, expr.measure, expr.unit),
            scope)


@dispatch(Map)
def _print_python(expr, leaves=None):
    child, scope = print_python(leaves, expr._child)
    funcname = next(funcnames)
    return ('%s(%s)' % (funcname, child),
            toolz.assoc(scope, funcname, expr.func))


@dispatch(notnull)
def _print_python(expr, leaves=None):
    child, scope = print_python(leaves, expr._child)
    return ('notnull(%s)' % child,
            toolz.merge(scope, dict(notnull=lambda x: x is not None)))


@dispatch(Like)
def _print_python(expr, leaves):
    child, scope = print_python(leaves, expr._child)
    return (
        'fnmatch(%s, %r)' % (child, expr.pattern),
        toolz.merge(scope, dict(fnmatch=fnmatch.fnmatch))
    )


@dispatch(Expr)
def _print_python(expr, leaves=None):
    raise NotImplementedError("Do not know how to write expressions of type %s"
                              " to Python code" % type(expr).__name__)


def funcstr(leaves, expr):
    """ Lambda string for an expresion

    >>> t = symbol('t', '{x: int, y: int, z: int, when: datetime}')

    >>> funcstr([t], t.x + t.y)
    ('lambda t: t[0] + t[1]', {})

    >>> funcstr([t.x, t.y], t.x + t.y)
    ('lambda x, y: x + y', {})

    Also returns scope for libraries like math or datetime

    >>> funcstr([t.x, t.y], sin(t.x) + t.y)  # doctest: +SKIP
    ('lambda x, y: math.sin(x) + y', {'math': <module 'math'>})

    >>> from datetime import date
    >>> funcstr([t.x, t.y, t.when], t.when.date > date(2001, 12, 25)) #doctest: +SKIP
    ('lambda x, y, when: when.day > datetime.date(2001, 12, 25)', {'datetime': <module 'datetime'>})
    """
    result, scope = print_python(leaves, expr)

    leaf_names = [print_python([leaf], leaf)[0] for leaf in leaves]

    return 'lambda %s: %s' % (', '.join(leaf_names),
                              result), scope


def lambdify(leaves, expr):
    """ Lambda for an expresion

    >>> t = symbol('t', '{x: int, y: int, z: int, when: datetime}')
    >>> f = lambdify([t], t.x + t.y)
    >>> f((1, 10, 100, ''))
    11

    >>> f = lambdify([t.x, t.y, t.z, t.when], t.x + cos(t.y))
    >>> f(1, 0, 100, '')
    2.0
    """
    s, scope = funcstr(leaves, expr)
    return eval(s, scope)
