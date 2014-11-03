from blaze.expr import *
from blaze.expr.collections import *
from toolz import isdistinct


t = Symbol('t', '5 * {name: string, amount: int, x: real}')

def test_merge():
    e = Symbol('e', '3 * 5 * {name: string, amount: int, x: real}')
    expr = merge(name=e.name, y=e.x)

    assert set(expr.fields) == set(['name', 'y'])
    assert expr.y.isidentical(e.x.label('y'))


def test_distinct():
    assert '5' not in str(t.distinct().dshape)


def test_join_on_same_columns():
    a = Symbol('a', 'var * {x: int, y: int, z: int}')
    b = Symbol('b', 'var * {x: int, y: int, w: int}')

    c = join(a, b, 'x')

    assert isdistinct(c.fields)
    assert len(c.fields) == 5
    assert 'b_y' in c.fields


def test_join_on_same_table():
    a = Symbol('a', 'var * {x: int, y: int}')

    c = join(a, a, 'x')

    assert isdistinct(c.fields)
    assert len(c.fields) == 3
