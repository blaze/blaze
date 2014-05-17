from blaze.expr.core import *


def test_subs():
    from blaze.expr.table import TableSymbol
    t = TableSymbol('{name: string, amount: int, id: int}')
    expr = t['amount'] + 3
    assert expr.subs({3: 4, 'amount': 'id'}).isidentical(
            t['id'] + 4)

    t2 = TableSymbol('{name: string, amount: int}')
    assert t['amount'].subs({t: t2}).isidentical(t2['amount'])
