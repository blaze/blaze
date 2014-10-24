from blaze.expr import *
from blaze.expr.broadcast2 import *
from blaze.expr.broadcast2 import (leaves_of_type, broadcast_collect,
        broadcast_table_collect, _table_find_leaves)
from blaze.compatibility import builtins
from toolz import isdistinct

x = Symbol('x', '5 * 3 * int32')
xx = Symbol('xx', 'int32')

y = Symbol('y', '5 * 3 * int32')
yy = Symbol('yy', 'int32')

a = Symbol('a', 'int32')


def test_broadcast_basic():
    b = Broadcast((x, y), (xx, yy), xx + yy)

    assert b.shape == x.shape
    assert b.schema == (xx + yy).dshape

    assert eval(str(b)).isidentical(b)


def test_scalar_symbols():
    exprs = [x, y]
    scalars = scalar_symbols(exprs)

    assert len(scalars) == len(exprs)
    assert isdistinct([s._name for s in scalars])
    assert builtins.all(s.dshape == e.schema for s, e in zip(scalars, exprs))


def test_broadcast_function():
    expr =  Pow(Add(x, Mult(2, y)), 2)  # (x + (2 * y)) ** 2
    b = broadcast(expr, [x, y])
    xx, yy = b._scalars
    assert b._scalar_expr.isidentical((xx + (2 * yy)) ** 2)

    # A different set of leaves
    b = broadcast(expr, [x, Mult(2, y)])
    xx, yy = b._scalars
    assert b._scalar_expr.isidentical((xx + yy) ** 2)


t = Symbol('t', 'var * {x: int, y: int, z: int}')


def test_tabular_case():
    expr = Add(x, Mult(y, 2))

    b = broadcast(expr, [t])
    tt, = b._scalars

    assert b._scalar_expr.isidentical(tt.x + tt.y * 2)


def test_optimize_broadcast():
    expr = (t.distinct().x + 1).distinct()

    expected = broadcast(t.distinct().x + 1, [t.distinct()]).distinct()
    result = broadcast_collect((Field, Arithmetic), expr)

    assert result.isidentical(expected)


def test_leaves_of_type():
    expr = Distinct(Distinct(Distinct(t.x)))

    result = leaves_of_type((Distinct,), expr)
    assert len(result) == 1
    assert list(result)[0].isidentical(t.x)


def test_broadcast_collect_doesnt_collect_scalars():
    expr = xx + yy * a

    assert broadcast_collect(Arithmetic, expr).isidentical(expr)


def test_table_broadcast():
    t = Symbol('t', 'var * {x: int, y: int, z: int}')

    expr = t.distinct()
    expr = (2 * expr.x + expr.y + 1).distinct()

    expected = t.distinct()
    expected = broadcast(2 * expected.x + expected.y + 1, [t]).distinct()
    assert broadcast_table_collect(expr).isidentical(expected)

    expr = (t.x + t.y).sum()
    result = broadcast_table_collect(expr)
    expected = broadcast(t.x + t.y, [t]).sum()
    assert result.isidentical(expected)


def test_table_broadcast_doesnt_affect_scalars():
    t = Symbol('t', '{x: int, y: int, z: int}')
    expr = (2 * t.x + t.y + 1)

    assert broadcast_table_collect(expr).isidentical(expr)


def test__table_find_leaves():
    t = Symbol('t', 'var * {x: int, y: int, z: int}')

    s = _table_find_leaves(t.x + t.y)
    assert isinstance(s, set)
    assert len(s) == 1
    assert first(s).isidentical(t)

    expr = t.distinct()
    s = _table_find_leaves(2 * expr.x + expr.y + 1)
    assert isinstance(s, set)
    assert len(s) == 1
    assert first(s).isidentical(t.distinct())


