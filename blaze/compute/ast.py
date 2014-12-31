from __future__ import absolute_import, division, print_function

from ..expr import Expr, Symbol, Field, Arithmetic, Math
from ..expr.expressions import valid_identifier
from ..expr.broadcast import broadcast_collect
from ..dispatch import dispatch
import datetime
import numpy as np
import toolz
import itertools
import ast


def make(cls, *args, **kwargs):
    kwargs.setdefault('lineno', 0)
    kwargs.setdefault('col_offset', 0)
    return cls(*args, **kwargs)


def variable(name, ctx=None):
    return make(ast.Name, valid_identifier(name), ast.Load())


funcnames = ('func_%d' % i for i in itertools.count())


def astify(leaves, expr):
    """ Print expression to be evaluated in Python

    >>> from blaze.expr import ceil, sin

    >>> t = symbol('t', '{x: int, y: int, z: int, when: datetime}')
    >>> astify([t], t.x + t.y)
    ('t[0] + t[1]', {})

    Supports mathematical and datetime access

    >>> astify([t], sin(t.x) > ceil(t.y))  # doctest: +SKIP
    ('math.sin(t[0]) > math.ceil(t[1])', {'math':<module 'math'>})
    >>> astify([t], t.when.day + 1)
    ('t[3].day + 1', {})

    Specify leaves of the expression to control level of printing

    >>> astify([t.x, t.y], t.x + t.y)
    ('x + y', {})

    Returns
    -------

    s: string
       A evalable string
    scope: dict
       A namespace to add to be given to eval
    """
    if isinstance(expr, Expr) and any(expr.isidentical(lf) for lf in leaves):
        return variable(expr._name), {}
    return _astify(expr, leaves=leaves)


@dispatch((datetime.datetime, datetime.date))
def _astify(expr, leaves=None):
    return ast.parse(repr(expr), mode='eval').body, {'datetime': datetime}


@dispatch(Symbol)
def _astify(expr, leaves=None):
    return variable(expr._name), {}


@dispatch(Field)
def _astify(expr, leaves=None):
    child, scope = astify(leaves, expr._child)
    index = make(ast.Index, make(ast.Str, expr._name))
    sub = make(ast.Subscript, child, index, ast.Load())
    return sub, scope


@dispatch(Arithmetic)
def _astify(expr, leaves=None):
    lhs, left_scope = astify(leaves, expr.lhs)
    rhs, right_scope = astify(leaves, expr.rhs)
    scope = toolz.merge(left_scope, right_scope)
    binop = make(ast.BinOp, lhs, binops[expr.symbol](), rhs)
    return binop, scope


binops = {
    '+': ast.Add,
    '-': ast.Sub,
    '*': ast.Mult,
    '/': ast.Div,
    '//': ast.FloorDiv,
    '**': ast.Pow,
    '%': ast.Mod,
}


@dispatch(Math)
def _astify(expr, leaves=None):
    child, scope = astify(leaves, expr._child)
    attr = make(ast.Attribute, variable('np'), expr.symbol, ast.Load())
    call = make(ast.Call, func=attr, args=[child], keywords=[], starargs=None,
                kwargs=None)
    return call, toolz.merge(scope, {'np': np})


@dispatch(Expr)
def _astify(expr, leaves=None):
    raise NotImplementedError("Do not know how to write expressions of type %s"
                              " to a Python AST" % type(expr).__name__)


def build_lambda(leaves, expr):
    body, scope = astify(leaves, expr)
    argnames = [variable(leaf._name) for leaf in leaves]
    args = make(ast.arguments, args=argnames, defaults=[])
    return make(ast.Lambda, args=args, body=body), scope


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
    body, scope = build_lambda(leaves, expr)
    eval_expr = make(ast.Expression, body=body)
    code_object = compile(eval_expr, filename=__file__, mode='eval')
    return eval(code_object, scope)
