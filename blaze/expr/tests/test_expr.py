from __future__ import absolute_import, division, print_function
from datashape import dshape

from blaze.expr.core import *
from blaze.expr.expr import *

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
