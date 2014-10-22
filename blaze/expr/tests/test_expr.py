from __future__ import absolute_import, division, print_function
from datashape import dshape

from blaze.expr import *

def test_Symbol():
    e = Symbol('e', '3 * 5 * {name: string, amount: int}')
    assert e.dshape == dshape('3 * 5 * {name: string, amount: int}')
    assert e.shape == (3, 5)
    assert str(e) == 'e'

def test_Field():
    e = Symbol('e', '3 * 5 * {name: string, amount: int}')
    assert 'name' in dir(e)
    assert e.name.dshape == dshape('3 * 5 * string')
    assert e.name.schema == dshape('string')
    assert e.amount._name == 'amount'


def test_nested_fields():
    e = Symbol('e', '3 * {name: string, payments: var * {amount: int, when: datetime}}')
    assert e.payments.dshape == dshape('3 * var * {amount: int, when: datetime}')
    assert e.payments.schema == dshape('{amount: int, when: datetime}')
    assert 'amount' in dir(e.payments)
    assert e.payments.amount.dshape == dshape('3 * var * int')

def test_partialed_methods_have_docstrings():
    e = Symbol('e', '3 * 5 * {name: string, amount: int}')
    assert 'string comparison' in e.like.__doc__


def test_relabel():
    e = Symbol('e', '{name: string, amount: int}')
    assert e.relabel(amount='balance').fields == ['name', 'balance']


def test_dir():
    e = Symbol('e', '3 * 5 * {name: string, amount: int, x: real}')

    assert 'name' in dir(e)
    assert 'name' not in dir(e.name)
    assert 'isnan' in dir(e.x)
    assert 'isnan' not in dir(e.amount)


def test_label():
    e = Symbol('e', '3 * int')
    assert e._name == 'e'
    assert label(e, 'foo')._name == 'foo'
    assert label(e, 'e').isidentical(e)


def test_fields_with_spaces():
    e = Symbol('e', '{x: int, "a b": int}')
    assert isinstance(e['a b'], Field)
    assert 'a b' not in dir(e)
