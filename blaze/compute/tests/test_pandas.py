from __future__ import absolute_import, division, print_function

from blaze.compute.tpandas import *
from blaze.expr.table import *
from pandas import DataFrame

t = TableExpr('{name: string, amount: int, id: int}')


df = DataFrame([['Alice', 100, 1],
                ['Bob', 200, 2],
                ['Alice', 50, 3]], columns=['name', 'amount', 'id'])


def test_table():
    assert str(compute(t, df)) == str(df)


def test_projection():
    assert str(compute(t['name'], df)) == str(df['name'])


def test_eq():
    assert ((compute(t['amount'] == 100, df))
             == (df['amount'] == 100)).all()


def test_selection():
    assert str(compute(t[t['amount'] == 0], df)) == str(df[df['amount'] == 0])
    assert str(compute(t[t['amount'] > 150], df)) == str(df[df['amount'] > 150])


def test_arithmetic():
    assert str(compute(t['amount'] + t['id'], df)) == \
                str(df.amount + df.id)
    assert str(compute(t['amount'] * t['id'], df)) == \
                str(df.amount * df.id)
    assert str(compute(t['amount'] % t['id'], df)) == \
                str(df.amount % df.id)

def test_join():
    left = DataFrame([['Alice', 100], ['Bob', 200]], columns=['name', 'amount'])
    right = DataFrame([['Alice', 1], ['Bob', 2]], columns=['name', 'id'])

    L = TableExpr('{name: string, amount: int}')
    R = TableExpr('{name: string, id: int}')
    joined = Join(L, R, 'name')

    assert dshape(joined.schema) == \
            dshape('{name: string, amount: int, id: int}')

    result = compute(joined, {L: left, R: right})

    expected = DataFrame([['Alice', 100, 1], ['Bob', 200, 2]],
                         columns=['name', 'amount', 'id'])

    assert str(result.reset_index()) == str(expected)

