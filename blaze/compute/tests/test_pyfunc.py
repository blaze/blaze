from blaze.compute.pyfunc import *
import datetime

t = Symbol('t', '{x: int, y: int, z: int, when: datetime}')

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
    f = lambdify([t], t.x + t.y.map(inc))
    assert f((1, 2, 3, 4)) == 1 + inc(2)


def test_math():
    f = lambdify([t], t.x + cos(t.y))
    assert f((1, 0, 3, 4)) == 1 + math.cos(0.0)


def test_datetime_literals():
    f = lambdify([t], t.when > '2000-01-01')
    assert f((1, 0, 3, datetime.datetime(2000, 1, 2))) == True
    assert f((1, 0, 3, datetime.datetime(1999, 1, 2))) == False


def test_broadcast_collect():
    t = Symbol('t', 'var * {x: int, y: int, z: int, when: datetime}')

    expr = t.distinct()
    expr = expr.x + 2*expr.y
    expr = expr.distinct()

    result = broadcast_collect(expr)

    expected = t.distinct()
    expected = broadcast(expected.x + 2*expected.y, [expected])
    expected = expected.distinct()

    assert result.isidentical(expected)
