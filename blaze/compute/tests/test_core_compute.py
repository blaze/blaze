from __future__ import absolute_import, division, print_function

from datetime import date, datetime

from blaze.compute.core import compute_up, compute_down, optimize, compute
from blaze import by, Symbol, Expr
from blaze.dispatch import dispatch
from blaze.compatibility import raises

def test_errors():
    t = Symbol('t', 'var * {foo: int}')
    with raises(NotImplementedError):
        compute_up(by(t, t.count()), 1)


def test_optimize():
    class Foo(object):
        pass

    s = Symbol('s', '5 * {x: int, y: int}')

    @dispatch(Expr, Foo)
    def compute_down(expr, foo):
        return str(expr)

    assert compute(s.x * 2, Foo()) == "s['x'] * 2"

    @dispatch(Expr, Foo)
    def optimize(expr, foo):
        return expr + 1

    assert compute(s.x * 2, Foo()) == "(s['x'] * 2) + 1"
