from __future__ import absolute_import, division, print_function

from datetime import date, datetime

from blaze.compute.core import (compute_up, compute_down, optimize, compute,
        bottom_up_until_type_break, top_then_bottom_then_top_again_etc,
        swap_resources_into_scope)
from blaze.expr import by, symbol, Expr, Symbol
from blaze.dispatch import dispatch
from blaze.compatibility import raises

import numpy as np


def test_errors():
    t = symbol('t', 'var * {foo: int}')
    with raises(NotImplementedError):
        compute_up(by(t, count=t.count()), 1)


def test_optimize():
    class Foo(object):
        pass

    s = symbol('s', '5 * {x: int, y: int}')

    @dispatch(Expr, Foo)
    def compute_down(expr, foo):
        return str(expr)

    assert compute(s.x * 2, Foo()) == "s.x * 2"

    @dispatch(Expr, Foo)
    def optimize(expr, foo):
        return expr + 1

    assert compute(s.x * 2, Foo()) == "(s.x * 2) + 1"


def test_bottom_up_until_type_break():

    s = symbol('s', 'var * {name: string, amount: int}')
    data = np.array([('Alice', 100), ('Bob', 200), ('Charlie', 300)],
                    dtype=[('name', 'S7'), ('amount', 'i4')])

    e = (s.amount + 1).distinct()
    expr, scope = bottom_up_until_type_break(e, {s: data})
    amount = symbol('amount', 'var * real', token=1)
    assert expr.isidentical(amount)
    assert len(scope) == 1
    assert amount in scope
    assert (scope[amount] == np.array([101, 201, 301], dtype='i4')).all()

    # This computation has a type change midstream, so we stop and get the
    # unfinished computation.

    e = s.amount.sum() + 1
    expr, scope = bottom_up_until_type_break(e, {s: data})
    amount_sum = symbol('amount_sum', 'int')
    assert expr.isidentical(amount_sum + 1)
    assert len(scope) == 1
    assert amount_sum in scope
    assert scope[amount_sum] == 600

    # ensure that we work on binops with one child
    x = symbol('x', 'real')
    expr, scope = bottom_up_until_type_break(x + x, {x: 1})
    assert len(scope) == 1
    x2 = list(scope.keys())[0]
    assert isinstance(x2, Symbol)
    assert isinstance(expr, Symbol)
    assert scope[x2] == 2


def test_top_then_bottom_then_top_again_etc():
    s = symbol('s', 'var * {name: string, amount: int}')
    data = np.array([('Alice', 100), ('Bob', 200), ('Charlie', 300)],
                    dtype=[('name', 'S7'), ('amount', 'i4')])

    e = s.amount.sum() + 1
    assert top_then_bottom_then_top_again_etc(e, {s: data}) == 601


def test_swap_resources_into_scope():

    from blaze import Data
    t = Data([1, 2, 3], dshape='3 * int', name='t')
    expr, scope = swap_resources_into_scope(t.head(2), {t: t.data})

    assert t._resources()
    assert not expr._resources()

    assert t not in scope
