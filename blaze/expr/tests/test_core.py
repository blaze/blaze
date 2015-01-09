from __future__ import absolute_import, division, print_function

from datashape import dshape
from blaze.expr import *
from blaze.expr.core import subs


def test_subs():
    from blaze.expr import TableSymbol
    t = TableSymbol('t', '{name: string, amount: int, id: int}')
    expr = t['amount'] + 3
    assert expr._subs({3: 4, 'amount': 'id'}).isidentical(
            t['id'] + 4)

    t2 = TableSymbol('t', '{name: string, amount: int}')
    assert t['amount']._subs({t: t2}).isidentical(t2['amount'])


def test_contains():
    from blaze.expr import TableSymbol
    t = TableSymbol('t', '{name: string, amount: int, id: int}')

    assert t in t['name']
    assert t in t['name'].distinct()
    assert t['id'] not in t['name']

    assert t['id'] in t['id'].sum()


def test_path():
    from blaze.expr import TableSymbol, join
    t = TableSymbol('t', '{name: string, amount: int, id: int}')
    v = TableSymbol('v', '{city: string, id: int}')
    expr = t['amount'].sum()

    assert list(path(expr, t)) == [t.amount.sum(), t.amount, t]
    assert list(path(expr, t.amount)) == [t.amount.sum(), t.amount]
    assert list(path(expr, t.amount)) == [t.amount.sum(), t.amount]

    expr = join(t, v).amount
    assert list(path(expr, t)) == [join(t, v).amount, join(t, v), t]
    assert list(path(expr, v)) == [join(t, v).amount, join(t, v), v]


def test_hash():
    e = symbol('e', 'int')
    assert '_hash' in e.__slots__
    assert not hasattr(e, '_hash')
    h = hash(e)
    assert isinstance(h, int)
    assert h == hash(e)

    assert hash(Symbol('e', 'int')) == hash(Symbol('e', 'int'))

    f = symbol('f', 'int')
    assert hash(e) != hash(f)

    assert hash(e._subs({'e': 'f'})) != hash(e)
    assert hash(e._subs({'e': 'f'})) == hash(f)

"""

def test_subs_on_datashape():
    assert subs(dshape('3 * {foo: int}'), {'foo': 'bar'}) == dshape('3 * {bar: int}')
"""
