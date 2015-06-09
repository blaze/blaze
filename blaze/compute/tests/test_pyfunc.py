import datetime

from blaze.compute.pyfunc import symbol, lambdify, cos, math, broadcast
from blaze.compute.pyfunc import _print_python
from blaze.expr.broadcast import broadcast_collect


t = symbol('t', '{x: int, y: int, z: int, when: datetime}')


def test_simple():
    f = lambdify([t], t.x + t.y)
    assert f((1, 2, 3, 4)) == 1 + 2

    f = lambdify([t.x, t.y], t.x + t.y)
    assert f(1, 2) == 1 + 2


def test_datetime():
    f = lambdify([t], t.x + t.when.year)

    assert f((1, 2, 3, datetime.datetime(2000, 1, 1))) == 1 + 2000


def inc(x):
    return x + 1


def test_map():
    f = lambdify([t], t.x + t.y.map(inc, 'int'))
    assert f((1, 2, 3, 4)) == 1 + inc(2)


def test_math():
    f = lambdify([t], abs(t.x) + cos(t.y))
    assert f((-1, 0, 3, 4)) == 1 + math.cos(0.0)


def test_datetime_literals_and__print_python():
    _print_python(datetime.date(2000, 1, 1)) == \
            'datetime.date(2000, 1, 1)', {'datetime': datetime}


def test_datetime_literals():
    f = lambdify([t], t.when > '2000-01-01')
    assert f((1, 0, 3, datetime.datetime(2000, 1, 2)))
    assert not f((1, 0, 3, datetime.datetime(1999, 1, 2)))


def test_broadcast_collect():
    t = symbol('t', 'var * {x: int, y: int, z: int, when: datetime}')

    expr = t.distinct()
    expr = expr.x + 2 * expr.y
    expr = expr.distinct()

    result = broadcast_collect(expr)

    expected = t.distinct()
    expected = broadcast(expected.x + 2 * expected.y, [expected])
    expected = expected.distinct()

    assert result.isidentical(expected)


def test_pyfunc_works_with_invalid_python_names():
    x = symbol('x-y.z', 'int')
    f = lambdify([x], x + 1)
    assert f(1) == 2

    t = symbol('t', '{"x.y": int, "y z": int}')
    f = lambdify([t], t.x_y + t.y_z)
    assert f((1, 2)) == 3


def test_usub():
    x = symbol('x', 'float64')
    f = lambdify([x], -x)
    assert f(1.0) == -1.0


def test_not():
    x = symbol('x', 'bool')
    f = lambdify([x], ~x)
    r = f(True)
    assert isinstance(r, bool) and not r

    r = f(False)
    assert isinstance(r, bool) and r
