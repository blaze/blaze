from __future__ import absolute_import, division, print_function

from datashape import dshape
from blaze.expr.core import *


def test_subs():
    from blaze.expr.table import TableSymbol
    t = TableSymbol('t', '{name: string, amount: int, id: int}')
    expr = t['amount'] + 3
    assert expr.subs({3: 4, 'amount': 'id'}).isidentical(
            t['id'] + 4)

    t2 = TableSymbol('t', '{name: string, amount: int}')
    assert t['amount'].subs({t: t2}).isidentical(t2['amount'])


def test_contains():
    from blaze.expr.table import TableSymbol, By
    t = TableSymbol('t', '{name: string, amount: int, id: int}')

    assert t in t['name']
    assert t in t['name'].distinct()
    assert t['id'] not in t['name']

    assert t['id'] in t['id'].sum()

def test_path():
    from blaze.expr.table import TableSymbol, join
    t = TableSymbol('t', '{name: string, amount: int, id: int}')
    v = TableSymbol('v', '{city: string, id: int}')
    expr = t['amount'].sum()

    assert list(path(expr, t)) == [t.amount.sum(), t.amount, t]
    assert list(path(expr, t.amount)) == [t.amount.sum(), t.amount]
    assert list(path(expr, t.amount)) == [t.amount.sum(), t.amount]

    expr = join(t, v).amount
    assert list(path(expr, t)) == [join(t, v).amount, join(t, v), t]
    assert list(path(expr, v)) == [join(t, v).amount, join(t, v), v]

def test_Symbol():
    e = ExprSymbol('e', '3 * 5 * {name: string, amount: int}')
    assert e.dshape == dshape('3 * 5 * {name: string, amount: int}')
    assert e.shape == (3, 5)
