from __future__ import absolute_import, division, print_function
from datashape import dshape

from blaze.expr import *
from blaze.expr.core import subs
from blaze.utils import raises


def test_Symbol():
    e = symbol('e', '3 * 5 * {name: string, amount: int}')
    assert e.dshape == dshape('3 * 5 * {name: string, amount: int}')
    assert e.shape == (3, 5)
    assert str(e) == 'e'

def test_symbol_caches():
    assert symbol('e', 'int') is symbol('e', 'int')

def test_Symbol_tokens():
    assert symbol('x', 'int').isidentical(Symbol('x', 'int'))
    assert not symbol('x', 'int').isidentical(Symbol('x', 'int', 1))

def test_Field():
    e = symbol('e', '3 * 5 * {name: string, amount: int}')
    assert 'name' in dir(e)
    assert e.name.dshape == dshape('3 * 5 * string')
    assert e.name.schema == dshape('string')
    assert e.amount._name == 'amount'


def test_nested_fields():
    e = symbol('e', '3 * {name: string, payments: var * {amount: int, when: datetime}}')
    assert e.payments.dshape == dshape('3 * var * {amount: int, when: datetime}')
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
    ts = t.timestamp
    assert raises(ValueError, lambda: ts.relabel({'timestamp': 'date',
                                                  'hello': 'world'}))


def test_map_with_rename():
    t = symbol('s', 'var * {timestamp: datetime}')
    result = t.timestamp.map(lambda x: x.date(), schema='{date: datetime}')
    assert raises(ValueError, lambda: result.relabel({'timestamp': 'date'}))
    assert result.fields == ['date']
