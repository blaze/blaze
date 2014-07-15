from __future__ import absolute_import, division, print_function

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
