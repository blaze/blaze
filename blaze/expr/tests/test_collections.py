from blaze.expr import *
from blaze.expr.collections import *
from blaze.utils import raises
from blaze.compatibility import builtins
from toolz import isdistinct


t = symbol('t', '5 * {name: string, amount: int, x: real}')

def test_merge():
    e = symbol('e', '3 * 5 * {name: string, amount: int, x: real}')
    expr = merge(name=e.name, y=e.x)

    assert set(expr.fields) == set(['name', 'y'])
    assert expr.y.isidentical(e.x.label('y'))


def test_merge_on_single_argument_is_noop():
    assert merge(t.name).isidentical(t.name)


def test_transform():
    t = symbol('t', '5 * {x: int, y: int}')
    expr = transform(t, z=t.x + t.y)
    assert expr.fields == ['x', 'y', 'z']

    assert builtins.any((t.x + t.y).isidentical(node)
                for node in expr._subterms())


def test_distinct():
    assert '5' not in str(t.distinct().dshape)


def test_join_on_same_columns():
    a = symbol('a', 'var * {x: int, y: int, z: int}')
    b = symbol('b', 'var * {x: int, y: int, w: int}')

    c = join(a, b, 'x')

    assert isdistinct(c.fields)
    assert len(c.fields) == 5
    assert 'y_left' in c.fields
    assert 'y_right' in c.fields


def test_join_on_same_table():
    a = symbol('a', 'var * {x: int, y: int}')

    c = join(a, a, 'x')

    assert isdistinct(c.fields)
    assert len(c.fields) == 3


def test_join_on_single_column():
    a = symbol('a', 'var * {x: int, y: int, z: int}')
    b = symbol('b', 'var * {x: int, y: int, w: int}')

    expr = join(a, b.x)

    assert expr.on_right == 'x'


def test_raise_error_if_join_on_no_columns():
    a = symbol('a', 'var * {x: int}')
    b = symbol('b', 'var * {y: int}')

    assert raises(ValueError, lambda: join(a, b))
