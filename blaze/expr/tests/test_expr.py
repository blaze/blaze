from __future__ import absolute_import, division, print_function

import types

import pandas as pd

import pytest

import datashape
from datashape import dshape, var, datetime_, float32, int64, bool_

from blaze.expr import symbol, label, Field


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
    assert 'string comparison' in e.like.__doc__


def test_relabel():
    e = symbol('e', '{name: string, amount: int}')
    assert e.relabel(amount='balance').fields == ['name', 'balance']


def test_meaningless_relabel_doesnt_change_input():
    e = symbol('e', '{name: string, amount: int}')
    assert e.relabel(amount='amount').isidentical(e)


def test_relabel_with_invalid_identifiers_reprs_as_dict():
    s = symbol('s', '{"0": int64}')
    assert repr(s.relabel({'0': 'foo'})) == "s.relabel({'0': 'foo'})"


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

    from blaze.expr.expressions import _attr_cache
    assert (expr, '_and') in _attr_cache
    assert (expr2, '_and') in _attr_cache


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
