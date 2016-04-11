import pytest

from datashape import dshape
from datashape.util.testing import assert_dshape_equal

from blaze.expr import symbol
from blaze.expr.core import common_subexpression
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


def test_merge_options():
    s = symbol('s', 'var * {a: ?A, b: ?B}')

    merged = merge(a=s.a, b=s.b)
    assert_dshape_equal(merged.dshape, dshape('var * {a: ?A, b: ?B}'))
    assert_dshape_equal(merged.a.dshape, dshape('var * ?A'))
    assert_dshape_equal(merged.b.dshape, dshape('var * ?B'))


def test_merge_on_single_argument_is_noop():
    assert merge(t.name).isidentical(t.name)


def test_merge_common_subexpression():
    t = symbol('t', 'var * {a: float64}')
    result = common_subexpression(t.a - t.a % 3, t.a % 3)
    assert result.isidentical(t.a)


def test_merge_exceptions():
    t = symbol('t', 'var * {x:int}')
    with pytest.raises(ValueError) as excinfo:
        merge(t, x=2*t.x)

    assert "Repeated columns" in str(excinfo.value)


def test_transform():
    t = symbol('t', '5 * {x: int, y: int}')
    expr = transform(t, z=t.x + t.y)
    assert expr.fields == ['x', 'y', 'z']

    assert builtins.any((t.x + t.y).isidentical(node)
                        for node in expr._subterms())


def test_distinct():
    assert '5' not in str(t.distinct().dshape)


def test_distinct_exceptions():
    t = symbol('t', 'var * {x:int}')
    with pytest.raises(ValueError) as excinfo:
        t.distinct('a')

    assert "a is not a field of t" == str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        t.distinct(10)

    assert "on must be a name or field, not: 10" == str(excinfo.value)

    s = symbol('s', 'var * {y:int}')
    with pytest.raises(ValueError) as excinfo:
        t.distinct(s.y)

    assert "s.y is not a field of t" == str(excinfo.value)


def test_sample_exceptions():
    t = symbol('t', 'var * {x:int, y:int}')
    with pytest.raises(ValueError):
        t.sample(n=1, frac=0.1)
    with pytest.raises(TypeError):
        t.sample(n='a')
    with pytest.raises(ValueError):
        t.sample(frac='a')
    with pytest.raises(TypeError):
        t.sample(foo='a')
    with pytest.raises(TypeError):
        t.sample()


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


@pytest.mark.xfail
def test_join_option_types_outer():
    a = symbol('a', 'var * {x: ?int}')
    b = symbol('b', 'var * {x: int}')

    assert (join(a, b, 'x', how='outer').dshape ==
            join(b, a, 'x', how='outer').dshape ==
            dshape('var * {x: ?int}'))


def test_join_option_string_types():
    a = symbol('a', 'var * {x: ?string}')
    b = symbol('b', 'var * {x: string}')
    c = symbol('c', 'var * {x: ?string}')

    assert (join(a, b, 'x').dshape ==
            join(b, a, 'x').dshape ==
            dshape('var * {x: string}'))

    assert (join(a, c, 'x').dshape ==
            join(c, a, 'x').dshape ==
            dshape('var * {x: ?string}'))


def test_join_exceptions():
    """
    exception raised for mismatched schema;
    exception raised for no shared fields
    """
    a = symbol('a', 'var * {x: int}')
    b = symbol('b', 'var * {x: string}')
    with pytest.raises(TypeError) as excinfo:
        join(a, b, 'x')

    assert "Schemata of joining columns do not match," in str(excinfo.value)
    assert "x=int32 and x=string" in str(excinfo.value)

    b = symbol('b', 'var * {z: int}')
    with pytest.raises(ValueError) as excinfo:
        join(a, b)

    assert "No shared columns between a and b" in str(excinfo.value)

    b = symbol('b', 'var * {x: int}')
    with pytest.raises(ValueError) as excinfo:
        join(a, b, how='inner_')

    assert "Got: inner_" in str(excinfo.value)


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

    with pytest.raises(TypeError) as excinfo:
        concat(a, b)

    msg = 'Mismatched measures: {l} != {r}'.format(l=a.dshape.measure,
                                                   r=b.dshape.measure)
    assert msg == str(excinfo.value)


def test_concat_different_along_concat_axis():
    a = symbol('a', '3 * 5 * int32')
    b = symbol('b', '3 * 6 * int32')

    with pytest.raises(TypeError) as excinfo:
        concat(a, b, axis=0)

    assert "not equal along axis 1: 5 != 6" in str(excinfo.value)

    b = symbol('b', '4 * 6 * int32')
    with pytest.raises(TypeError) as excinfo:
        concat(a, b, axis=1)

    assert "not equal along axis 0: 3 != 4" in str(excinfo.value)


def test_concat_negative_axis():
    a = symbol('a', '3 * 5 * int32')
    b = symbol('b', '3 * 5 * int32')

    with pytest.raises(ValueError) as excinfo:
        concat(a, b, axis=-1)

    assert "must be in range: [0, 2)" in str(excinfo.value)


def test_concat_axis_too_great():
    a = symbol('a', '3 * 5 * int32')
    b = symbol('b', '3 * 5 * int32')

    with pytest.raises(ValueError) as excinfo:
        concat(a, b, axis=2)

    assert "must be in range: [0, 2)" in str(excinfo.value)


def test_shift():
    t = symbol('t', 'var * float64')
    assert t.shift(1).dshape == dshape('var * ?float64')
    assert t.shift(0).dshape == t.dshape
    assert t.shift(-1).dshape == dshape('var * ?float64')
    assert str(t.shift(1)) == 'shift(t, n=1)'
    assert str(t.shift(0)) == 'shift(t, n=0)'
    assert str(t.shift(-1)) == 'shift(t, n=-1)'


def test_shift_not_int():
    a = symbol('a', '3 * {x: int32, y: int32}')
    with pytest.raises(TypeError):
        a.x.shift(1.3)
