import pytest

from datashape import dshape

from blaze.expr import symbol
from blaze.expr.collections import merge, join, transform, concat
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


def test_join_suffixes():
    a = symbol('a', 'var * {x: int, y: int}')
    b = join(a, a, 'x', suffixes=('_l', '_r'))

    assert isdistinct(b.fields)
    assert len(b.fields) == 3
    assert set(b.fields) == set(['x', 'y_l', 'y_r'])


def test_join_on_single_column():
    a = symbol('a', 'var * {x: int, y: int, z: int}')
    b = symbol('b', 'var * {x: int, y: int, w: int}')

    expr = join(a, b.x)

    assert expr.on_right == 'x'


def test_raise_error_if_join_on_no_columns():
    a = symbol('a', 'var * {x: int}')
    b = symbol('b', 'var * {y: int}')

    assert raises(ValueError, lambda: join(a, b))


def test_join_option_types():
    a = symbol('a', 'var * {x: ?int}')
    b = symbol('b', 'var * {x: int}')

    assert join(a, b, 'x').dshape == dshape('var * {x: int}')
    assert join(b, a, 'x').dshape == dshape('var * {x: int}')


def test_join_mismatched_schema():
    a = symbol('a', 'var * {x: int}')
    b = symbol('b', 'var * {x: string}')

    with pytest.raises(TypeError):
        join(a, b, 'x')


def test_join_type_promotion():
    a = symbol('a', 'var * {x: int32}')
    b = symbol('b', 'var * {x: int64}')

    assert join(a, b, 'x').dshape == dshape('var * {x: int64}')


def test_join_type_promotion_option():
    a = symbol('a', 'var * {x: ?int32}')
    b = symbol('b', 'var * {x: int64}')

    assert join(a, b, 'x').dshape == dshape('var * {x: int64}')


def test_isin():
    a = symbol('a', 'var * {x: int, y: string}')
    assert hasattr(a.x, 'isin')
    assert hasattr(a.y, 'isin')
    assert not hasattr(a, 'isin')


def test_isin_no_expressions():
    a = symbol('a', 'var * int')
    b = symbol('b', 'var * int')
    with pytest.raises(TypeError):
        a.isin(b)


def test_concat_table():
    a = symbol('a', '3 * {a: int32, b: int32}')
    b = symbol('a', '5 * {a: int32, b: int32}')
    v = symbol('v', 'var * {a: int32, b: int32}')

    assert concat(a, b).dshape == dshape('8 * {a: int32, b: int32}')
    assert concat(a, v).dshape == dshape('var * {a: int32, b: int32}')


def test_concat_mat():
    a = symbol('a', '3 * 5 * int32')
    b = symbol('b', '3 * 5 * int32')
    v = symbol('v', 'var * 5 * int32')
    u = symbol('u', '3 * var * int32')

    assert concat(a, b, axis=0).dshape == dshape('6 * 5 * int32')
    assert concat(a, b, axis=1).dshape == dshape('3 * 10 * int32')
    assert concat(a, v, axis=0).dshape == dshape('var * 5 * int32')
    assert concat(a, u, axis=1).dshape == dshape('3 * var * int32')


def test_concat_arr():
    a = symbol('a', '3 * int32')
    b = symbol('b', '5 * int32')
    v = symbol('v', 'var * int32')

    assert concat(a, b).dshape == dshape('8 * int32')
    assert concat(a, v).dshape == dshape('var * int32')


def test_concat_different_measure():
    a = symbol('a', '3 * 5 * int32')
    b = symbol('b', '3 * 5 * float64')

    with pytest.raises(TypeError):
        concat(a, b)


def test_concat_different_along_concat_axis():
    a = symbol('a', '3 * 5 * int32')
    b = symbol('b', '3 * 6 * int32')

    with pytest.raises(TypeError):
        concat(a, b, axis=0)


def test_concat_negative_axis():
    a = symbol('a', '3 * 5 * int32')
    b = symbol('b', '3 * 5 * int32')

    with pytest.raises(ValueError):
        concat(a, b, axis=-1)


def test_concat_axis_too_great():
    a = symbol('a', '3 * 5 * int32')
    b = symbol('b', '3 * 5 * int32')

    with pytest.raises(ValueError):
        concat(a, b, axis=2)
