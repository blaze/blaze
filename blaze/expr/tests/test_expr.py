from __future__ import absolute_import, division, print_function

import types

import datashape
from datashape import dshape, var, datetime_, float32, int64, bool_
from datashape.util.testing import assert_dshape_equal
import pandas as pd
import pytest

from blaze.compatibility import pickle
from blaze.expr import (
    Expr,
    Field,
    Node,
    coalesce,
    label,
    symbol,
    transform,
)


def test_slots():
    assert Expr.__slots__ == ('_hash', '__weakref__', '__dict__')
    assert Node.__slots__ == ()


def test_Symbol():
    e = symbol('e', '3 * 5 * {name: string, amount: int}')
    assert e.dshape == dshape('3 * 5 * {name: string, amount: int}')
    assert e.shape == (3, 5)
    assert str(e) == 'e'


def test_symbol_caches():
    assert symbol('e', 'int') is symbol('e', 'int')


def test_Symbol_tokens():
    assert symbol('x', 'int').isidentical(symbol('x', 'int'))
    assert not symbol('x', 'int').isidentical(symbol('x', 'int', 1))


def test_Field():
    e = symbol('e', '3 * 5 * {name: string, amount: int}')
    assert 'name' in dir(e)
    assert e.name.dshape == dshape('3 * 5 * string')
    assert e.name.schema == dshape('string')
    assert e.amount._name == 'amount'


def test_nested_fields():
    e = symbol(
        'e', '3 * {name: string, payments: var * {amount: int, when: datetime}}')
    assert e.payments.dshape == dshape(
        '3 * var * {amount: int, when: datetime}')
    assert e.payments.schema == dshape('{amount: int, when: datetime}')
    assert 'amount' in dir(e.payments)
    assert e.payments.amount.dshape == dshape('3 * var * int')


def test_partialed_methods_have_docstrings():
    e = symbol('e', '3 * 5 * {name: string, amount: int}')
    assert 'string comparison' in e.name.like.__doc__


def test_relabel():
    e = symbol('e', '{name: string, amount: int}')
    assert e.relabel(amount='balance').fields == ['name', 'balance']


def test_meaningless_relabel_doesnt_change_input():
    e = symbol('e', '{name: string, amount: int}')
    assert e.relabel(amount='amount').isidentical(e)


def test_relabel_with_invalid_identifiers_reprs_as_dict():
    s = symbol('s', '{"0": int64}')
    assert str(s.relabel({'0': 'foo'})) == "s.relabel({'0': 'foo'})"


def test_dir():
    e = symbol('e', '3 * 5 * {name: string, amount: int, x: real}')

    assert 'name' in dir(e)
    assert 'name' not in dir(e.name)
    assert 'isnan' in dir(e.x)
    assert 'isnan' not in dir(e.amount)


def test_label():
    e = symbol('e', '3 * int')
    assert e._name == 'e'
    assert label(e, 'foo')._name == 'foo'
    assert label(e, 'e').isidentical(e)


def test_fields_with_spaces():
    e = symbol('e', '{x: int, "a b": int}')
    assert isinstance(e['a b'], Field)
    assert 'a b' not in dir(e)

    assert 'a_b' in dir(e)
    assert e.a_b.isidentical(e['a b'])


def test_fields_with_spaces():
    e = symbol('e', '{x: int, "a.b": int}')
    assert isinstance(e['a.b'], Field)
    assert 'a.b' not in dir(e)

    assert 'a_b' in dir(e)
    assert e.a_b.isidentical(e['a.b'])


def test_selection_name_matches_child():
    t = symbol('t', 'var * {x: int, "a.b": int}')
    assert t.x[t.x > 0]._name == t.x._name
    assert t.x[t.x > 0].fields == t.x.fields


def test_symbol_subs():
    assert symbol('e', '{x: int}') is symbol('e', '{x: int}', None)
    assert symbol('e', '{x: int}') is symbol('e', dshape('{x: int}'))
    e = symbol('e', '{x: int, y: int}')
    f = symbol('f', '{x: int, y: int}')
    d = {'e': 'f'}
    assert e._subs(d) is f


def test_multiple_renames_on_series_fails():
    t = symbol('s', 'var * {timestamp: datetime}')
    with pytest.raises(ValueError):
        t.timestamp.relabel(timestamp='date', hello='world')


def test_map_with_rename():
    t = symbol('s', 'var * {timestamp: datetime}')
    result = t.timestamp.map(lambda x: x.date(), schema='{date: datetime}')
    with pytest.raises(ValueError):
        result.relabel(timestamp='date')
    assert result.fields == ['date']


def test_non_option_does_not_have_notnull():
    s = symbol('s', '5 * int32')
    assert not hasattr(s, 'notnull')


def test_notnull_dshape():
    assert symbol('s', '5 * ?int32').notnull().dshape == 5 * bool_


def test_hash_to_different_values():
    s = symbol('s', var * datetime_)
    expr = s >= pd.Timestamp('20121001')
    expr2 = s >= '20121001'
    assert expr2 & expr is not None
    assert hash(expr) == hash(expr2)


def test_hash():
    e = symbol('e', 'int')
    assert '_hash' in e.__slots__
    h = hash(e)
    assert isinstance(h, int)
    assert h == hash(e)

    assert hash(symbol('e', 'int')) == hash(symbol('e', 'int'))

    f = symbol('f', 'int')
    assert hash(e) != hash(f)

    assert hash(e._subs({'e': 'f'})) != hash(e)
    assert hash(e._subs({'e': 'f'})) == hash(f)


@pytest.mark.parametrize('dshape', [var * float32,
                                    dshape('var * float32'),
                                    'var * float32'])
def test_coerce(dshape):
    s = symbol('s', dshape)
    expr = s.coerce('int64')
    assert str(expr) == "s.coerce(to='int64')"
    assert expr.dshape == var * int64
    assert expr.schema == datashape.dshape('int64')
    assert expr.schema == expr.to


@pytest.mark.xfail(raises=AttributeError, reason='Should this be valid?')
def test_coerce_record():
    s = symbol('s', 'var * {a: int64, b: float64}')
    expr = s.coerce('{a: float64, b: float32}')
    assert str(expr) == "s.coerce(to='{a: float64, b: float32}')"


def test_method_before_name():
    t = symbol('t', 'var * {isin: int64, max: float64, count: int64}')
    assert isinstance(t['isin'], Field)
    assert isinstance(t['max'], Field)
    assert isinstance(t.max, Field)
    assert isinstance(t.isin, Field)
    assert isinstance(t['isin'].isin, types.MethodType)
    assert isinstance(t['max'].max, types.MethodType)
    assert isinstance(t.max.max, types.MethodType)
    assert isinstance(t.isin.isin, types.MethodType)
    with pytest.raises(AttributeError):
        t.count.max()


def test_pickle_roundtrip():
    t = symbol('t', 'var * int64')
    expr = (t + 1).mean()  # some expression with more than one node.
    assert expr.isidentical(pickle.loads(pickle.dumps(expr)))


def test_coalesce():
    # check case where lhs is not optional
    s = symbol('s', 'int32')
    t = symbol('t', 'int32')
    expr = coalesce(s, t)
    assert expr.isidentical(s)

    s_expr = s + s
    t_expr = t * 3
    expr = coalesce(s_expr, t_expr)
    assert expr.isidentical(s_expr)

    a = symbol('a', 'string')
    b = symbol('b', 'string')
    expr = coalesce(a, b)
    assert expr.isidentical(a)

    a_expr = a + a
    b_expr = b * 3
    expr = coalesce(a_expr, b_expr)
    assert expr.isidentical(a_expr)

    c = symbol('c', '{a: int32, b: int32}')
    d = symbol('d', '{a: int32, b: int32}')
    expr = coalesce(c, d)
    assert expr.isidentical(c)

    c_expr = transform(c, a=c.a + 1)
    d_expr = transform(d, a=d.a * 3)
    expr = coalesce(c_expr, d_expr)
    assert expr.isidentical(c_expr)

    # check case where lhs is null dshape
    u = symbol('u', 'null')
    expr = coalesce(u, s)
    assert expr.isidentical(s)

    expr = coalesce(u, a)
    assert expr.isidentical(a)

    expr = coalesce(u, c)
    assert expr.isidentical(c)

    # check optional lhs non-optional rhs
    v = symbol('v', '?int32')
    expr = coalesce(v, s)
    # rhs is not optional so the expression cannot be null
    assert_dshape_equal(expr.dshape, dshape('int32'))
    assert expr.lhs.isidentical(v)
    assert expr.rhs.isidentical(s)

    e = symbol('e', '?string')
    expr = coalesce(e, a)
    assert_dshape_equal(expr.dshape, dshape('string'))
    assert expr.lhs.isidentical(e)
    assert expr.rhs.isidentical(a)

    f = symbol('f', '?{a: int32, b: int32}')
    expr = coalesce(f, c)
    assert_dshape_equal(expr.dshape, dshape('{a: int32, b: int32}'))
    assert expr.lhs.isidentical(f)
    assert expr.rhs.isidentical(c)

    # check optional lhs non-optional rhs with promotion
    w = symbol('w', 'int64')
    expr = coalesce(v, w)
    # rhs is not optional so the expression cannot be null
    # there are no either types in datashape so we are a type large enough
    # to hold either result
    assert_dshape_equal(expr.dshape, dshape('int64'))
    assert expr.lhs.isidentical(v)
    assert expr.rhs.isidentical(w)

    # check optional lhs and rhs
    x = symbol('x', '?int32')
    expr = coalesce(v, x)
    # rhs and lhs are optional so this might be null
    assert_dshape_equal(expr.dshape, dshape('?int32'))
    assert expr.lhs.isidentical(v)
    assert expr.rhs.isidentical(x)

    # check optional lhs and rhs with promotion
    y = symbol('y', '?int64')
    expr = coalesce(v, y)
    # rhs and lhs are optional so this might be null
    # there are no either types in datashape so we are a type large enough
    # to hold either result
    assert_dshape_equal(expr.dshape, dshape('?int64'))
    assert expr.lhs.isidentical(v)
    assert expr.rhs.isidentical(y)


@pytest.mark.xfail(TypeError, reason='currently invalid type promotion')
@pytest.mark.parametrize('lhs,rhs,expected', (
    ('?{a: int32}', '{a: int64}', '{a: int64}'),
    ('?{a: int32}', '?{a: int64}', '?{a: int64}'),
))
def test_coalesce_invalid_promotion(lhs, rhs, expected):
    # Joe 2016-03-16: imho promote(record, record) should check that the keys
    # are the same and then create a new record from:
    # zip(keys, map(promote, lhs, rhs))
    f = symbol('e', lhs)
    g = symbol('g', rhs)
    expr = coalesce(f, g)
    assert_dshape_equal(expr.dshape, dshape(expected))
    assert expr.lhs.isidentical(f)
    assert expr.rhs.isidentical(g)


def test_cast():
    s = symbol('s', 'int32')

    assert_dshape_equal(s.cast('int64').dshape, dshape('int64'))
    assert_dshape_equal(s.cast(dshape('int64')).dshape, dshape('int64'))
    assert_dshape_equal(s.cast('var * int32').dshape, dshape('var * int32'))
    assert_dshape_equal(
        s.cast(dshape('var * int64')).dshape,
        dshape('var * int64'),
    )
    assert_dshape_equal(s.cast('var * int64').dshape, dshape('var * int64'))
    assert_dshape_equal(
        s.cast(dshape('var * int64')).dshape,
        dshape('var * int64'),
    )
