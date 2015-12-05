from blaze.expr import *
from blaze.expr.broadcast import *
from blaze.expr.broadcast import leaves_of_type, broadcast_collect
from blaze.compatibility import builtins
from toolz import isdistinct

x = symbol('x', '5 * 3 * int32')
xx = symbol('xx', 'int32')

y = symbol('y', '5 * 3 * int32')
yy = symbol('yy', 'int32')

a = symbol('a', 'int32')


def test_broadcast_basic():
    b = Broadcast((x, y), (xx, yy), xx + yy)

    assert b.shape == x.shape
    assert b.schema == (xx + yy).dshape

    assert eval(str(b)) is b


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
    assert b._scalar_expr is (xx + (2 * yy)) ** 2

    # A different set of leaves
    b = broadcast(expr, [x, Mult(2, y)])
    xx, yy = b._scalars
    assert b._scalar_expr is (xx + yy) ** 2


t = symbol('t', 'var * {x: int, y: int, z: int}')


def test_tabular_case():
    expr = t.x + t.y * 2

    b = broadcast(expr, [t])
    tt, = b._scalars

    assert b._scalar_expr is tt.x + tt.y * 2


def test_optimize_broadcast():
    expr = (t.distinct().x + 1).distinct()

    expected = broadcast(t.distinct().x + 1, [t.distinct()]).distinct()
    result = broadcast_collect(
        expr,
        broadcastable=(Field, Arithmetic),
        want_to_broadcast=(Field, Arithmetic),
    )

    assert result is expected


def test_leaves_of_type():
    expr = Distinct(Distinct(Distinct(t.x)))

    result = leaves_of_type((Distinct,), expr)
    assert len(result) == 1
    assert list(result)[0] is t.x


def test_broadcast_collect_doesnt_collect_scalars():
    expr = xx + yy * a

    assert broadcast_collect(
        expr,
        broadcastable=Arithmetic,
        want_to_broadcast=Arithmetic,
    ) is expr


def test_table_broadcast():
    t = symbol('t', 'var * {x: int, y: int, z: int}')

    expr = t.distinct()
    expr = (2 * expr.x + expr.y + 1).distinct()

    expected = t.distinct()
    expected = broadcast(2 * expected.x + expected.y + 1, [expected]).distinct()
    assert broadcast_collect(expr) is expected

    expr = (t.x + t.y).sum()
    result = broadcast_collect(expr)
    expected = broadcast(t.x + t.y, [t]).sum()
    assert result is expected


def test_broadcast_doesnt_affect_scalars():
    t = symbol('t', '{x: int, y: int, z: int}')
    expr = (2 * t.x + t.y + 1)

    assert broadcast_collect(expr) is expr


def test_full_expr():
    b = Broadcast((x, y), (xx, yy), xx + yy)
    assert b._full_expr is x + y


def test_broadcast_naming():
    t = symbol('t', 'var * {x: int, y: int, z: int}')

    for expr in [t.x, t.x + 1]:
        assert broadcast(expr, [t])._name == 'x'
