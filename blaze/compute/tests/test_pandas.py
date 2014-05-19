from __future__ import absolute_import, division, print_function

from blaze.compute.pandas import *
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

    print(result)
    print(expected)
    assert str(result) == str(expected)

    assert list(result.columns) == list(joined.columns)

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


def test_by_one():
    result = compute(By(t, t['name'], sum(t['amount'])), df)
    expected = df.groupby('name')['amount'].apply(lambda x: x.sum())

    assert str(result) == str(expected)


def test_by_two():
    result = compute(By(tbig, tbig[['name', 'sex']], sum(tbig['amount'])), dfbig)

    expected = dfbig.groupby(['name', 'sex'])['amount'].apply(lambda x: x.sum())

    assert str(result) == str(expected)


def test_by_three():
    result = compute(By(tbig,
                        tbig[['name', 'sex']],
                        (tbig['id'] + tbig['amount']).sum()),
                     dfbig)

    expected = dfbig.groupby(['name', 'sex']).apply(
            lambda df: (df['amount'] + df['id']).sum())

    assert str(result) == str(expected)

def test_by_four():
    t = tbig[['sex', 'amount']]
    result = compute(By(t, t['sex'], t['amount'].max()), dfbig)

    expected = dfbig[['sex', 'amount']].groupby('sex').apply(
            lambda df: df['amount'].max())

    assert str(result) == str(expected)


def test_join_by_arcs():
    df_idx = DataFrame([['A', 1],
                        ['B', 2],
                        ['C', 3]],
                      columns=['name', 'node_id'])

    df_arc = DataFrame([[1, 3],
                        [2, 3],
                        [3, 1]],
                       columns=['node_out', 'node_id'])

    t_idx = TableSymbol('{name: string, node_id: int32}')

    t_arc = TableSymbol('{node_out: int32, node_id: int32}')

    joined = Join(t_arc, t_idx, "node_id")

    want = By(joined, joined['name'], joined['node_id'].count())

    result = compute(want, {t_arc: df_arc, t_idx:df_idx})

    result_pandas = pandas.merge(df_arc, df_idx, on='node_id')

    assert str(result) == str(result_pandas.groupby('name')['node_id'].count())


def test_sort():
    print(str(compute(t.sort('amount'), df)))
    print(str(df.sort('amount')))
    assert str(compute(t.sort('amount'), df)) == str(df.sort('amount'))

    assert str(compute(t.sort('amount', ascending=True), df)) == \
            str(df.sort('amount', ascending=True))

    assert str(compute(t.sort(['amount', 'id']), df)) == \
            str(df.sort(['amount', 'id']))
