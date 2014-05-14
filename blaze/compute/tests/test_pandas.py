from __future__ import absolute_import, division, print_function

from blaze.compute.tpandas import *
from blaze.expr.table import *
from pandas import DataFrame

t = TableSymbol('{name: string, amount: int, id: int}')


df = DataFrame([['Alice', 100, 1],
                ['Bob', 200, 2],
                ['Alice', 50, 3]], columns=['name', 'amount', 'id'])


tbig = TableSymbol('{name: string, sex: string[1], amount: int, id: int}')

dfbig = DataFrame([['Alice', 'F', 100, 1],
                   ['Alice', 'F', 100, 3],
                   ['Drew', 'F', 100, 4],
                   ['Drew', 'M', 100, 5],
                   ['Drew', 'M', 200, 5]],
                  columns=['name', 'sex', 'amount', 'id'])


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

    L = TableSymbol('{name: string, amount: int}')
    R = TableSymbol('{name: string, id: int}')
    joined = Join(L, R, 'name')

    assert dshape(joined.schema) == \
            dshape('{name: string, amount: int, id: int}')

    result = compute(joined, {L: left, R: right})

    expected = DataFrame([['Alice', 100, 1], ['Bob', 200, 2]],
                         columns=['name', 'amount', 'id'])

    assert str(result.reset_index()) == str(expected)

def test_UnaryOp():
    assert (compute(exp(t['amount']), df) == np.exp(df['amount'])).all()


def test_columns_series():
    assert isinstance(compute(t['amount'], df), Series)
    assert isinstance(compute(t['amount'] > 150, df), Series)


def test_Reductions():
    assert compute(mean(t['amount']), df) == 350./3
    assert compute(count(t['amount']), df) == 3
    assert compute(sum(t['amount']), df) == 100 + 200 + 50
    assert compute(min(t['amount']), df) == 50
    assert compute(max(t['amount']), df) == 200
    assert compute(any(t['amount'] > 150), df) == True
    assert compute(any(t['amount'] > 250), df) == False


def test_by():
    result = compute(By(t, t['name'], sum(t['amount'])), df)
    expected = df.groupby('name')['amount'].apply(lambda x: x.sum())

    assert str(result) == str(expected)


def test_by_big():
    result = compute(By(tbig, tbig[['name', 'sex']], sum(tbig['amount'])), dfbig)

    expected = dfbig.groupby(['name', 'sex'])['amount'].apply(lambda x: x.sum())

    assert str(result) == str(expected)
