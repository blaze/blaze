from blaze.expr import *
from blaze.expr.collections import *


t = Symbol('t', '5 * {name: string, amount: int, x: real}')

def test_merge():
    e = Symbol('e', '3 * 5 * {name: string, amount: int, x: real}')
    expr = merge(name=e.name, y=e.x)

    assert set(expr.fields) == set(['name', 'y'])
    assert expr.y.isidentical(e.x.label('y'))


def test_distinct():
    assert '5' not in str(t.distinct().dshape)
