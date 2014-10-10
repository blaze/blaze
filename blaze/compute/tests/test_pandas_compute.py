from __future__ import absolute_import, division, print_function

import pytest

from datetime import datetime
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from blaze.compute.core import compute
from blaze import dshape, Table, discover, transform
from blaze.expr import TableSymbol, join, by, summary, Distinct
from blaze.expr import (merge, exp, mean, count, nunique, Apply, union, sum,
                        min, max, any, all, Projection, var, std)
from blaze.compatibility import builtins, xfail

t = TableSymbol('t', '{name: string, amount: int, id: int}')


df = DataFrame([['Alice', 100, 1],
                ['Bob', 200, 2],
                ['Alice', 50, 3]], columns=['name', 'amount', 'id'])


tbig = TableSymbol('tbig', '{name: string, sex: string[1], amount: int, id: int}')

dfbig = DataFrame([['Alice', 'F', 100, 1],
                   ['Alice', 'F', 100, 3],
                   ['Drew', 'F', 100, 4],
                   ['Drew', 'M', 100, 5],
                   ['Drew', 'M', 200, 5]],
                  columns=['name', 'sex', 'amount', 'id'])


def df_all(a_df, b_df):
    """Checks if two dataframes have the same columns

    This method doesn't check the index which can be manipulated during operations.
    """
    assert all(a_df.columns == b_df.columns)
    for col in a_df.columns:
        assert np.all(a_df[col] == b_df[col])
    return True


def test_series_columnwise():
    s = Series([1, 2, 3], name='a')
    t = TableSymbol('t', '{a: int64}')
    result = compute(t.a + 1, s)
    pd.util.testing.assert_series_equal(s + 1, result)


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

    L = TableSymbol('L', '{name: string, amount: int}')
    R = TableSymbol('R', '{name: string, id: int}')
    joined = join(L, R, 'name')

    assert (dshape(joined.schema) ==
            dshape('{name: string, amount: int, id: int}'))

    result = compute(joined, {L: left, R: right})

    expected = DataFrame([['Alice', 100, 1], ['Bob', 200, 2]],
                         columns=['name', 'amount', 'id'])

    print(result)
    print(expected)
    assert str(result) == str(expected)

    assert list(result.columns) == list(joined.fields)


def test_multi_column_join():
    left = [(1, 2, 3),
            (2, 3, 4),
            (1, 3, 5)]
    left = DataFrame(left, columns=['x', 'y', 'z'])
    right = [(1, 2, 30),
             (1, 3, 50),
             (1, 3, 150)]
    right = DataFrame(right, columns=['x', 'y', 'w'])

    L = TableSymbol('L', '{x: int, y: int, z: int}')
    R = TableSymbol('R', '{x: int, y: int, w: int}')

    j = join(L, R, ['x', 'y'])

    expected = [(1, 2, 3, 30),
                (1, 3, 5, 50),
                (1, 3, 5, 150)]
    expected = DataFrame(expected, columns=['x', 'y', 'z', 'w'])

    result = compute(j, {L: left, R: right})

    print(result)

    assert str(result) == str(expected)
    assert list(result.columns) == list(j.fields)


def test_unary_op():
    assert (compute(exp(t['amount']), df) == np.exp(df['amount'])).all()


def test_neg():
    assert (compute(-t['amount'], df) == -df['amount']).all()


@xfail(reason='Projection does not support arithmetic')
def test_neg_projection():
    assert (compute(-t[['amount', 'id']], df) == -df[['amount', 'id']]).all()


def test_columns_series():
    assert isinstance(compute(t['amount'], df), Series)
    assert isinstance(compute(t['amount'] > 150, df), Series)


def test_reductions():
    assert compute(mean(t['amount']), df) == 350./3
    assert compute(count(t['amount']), df) == 3
    assert compute(sum(t['amount']), df) == 100 + 200 + 50
    assert compute(min(t['amount']), df) == 50
    assert compute(max(t['amount']), df) == 200
    assert compute(nunique(t['amount']), df) == 3
    assert compute(nunique(t['name']), df) == 2
    assert compute(any(t['amount'] > 150), df) == True
    assert compute(any(t['amount'] > 250), df) == False
    assert compute(var(t['amount']), df) == df.amount.var(ddof=0)
    assert compute(var(t['amount'], unbiased=True), df) == df.amount.var()
    assert compute(std(t['amount']), df) == df.amount.std(ddof=0)
    assert compute(std(t['amount'], unbiased=True), df) == df.amount.std()


def test_distinct():
    dftoobig = DataFrame([['Alice', 'F', 100, 1],
                          ['Alice', 'F', 100, 1],
                          ['Alice', 'F', 100, 3],
                          ['Drew', 'F', 100, 4],
                          ['Drew', 'M', 100, 5],
                          ['Drew', 'F', 100, 4],
                          ['Drew', 'M', 100, 5],
                          ['Drew', 'M', 200, 5],
                          ['Drew', 'M', 200, 5]],
                          columns=['name', 'sex', 'amount', 'id'])
    d_t = Distinct(tbig)
    d_df = compute(d_t, dftoobig)
    assert df_all(d_df, dfbig)
    # Test idempotence
    assert df_all(compute(d_t, d_df), d_df)


def test_by_one():
    result = compute(by(t['name'], t['amount'].sum()), df)
    expected = df.groupby('name')['amount'].sum().reset_index()
    expected.columns = ['name', 'amount_sum']

    assert str(result) == str(expected)


def test_by_two():
    result = compute(by(tbig[['name', 'sex']], sum(tbig['amount'])), dfbig)

    expected = DataFrame([['Alice', 'F', 200],
                          ['Drew',  'F', 100],
                          ['Drew',  'M', 300]],
                          columns=['name', 'sex', 'amount_sum'])

    assert str(result) == str(expected)


def test_by_three():

    expr = by(tbig[['name', 'sex']],
              (tbig['id'] + tbig['amount']).sum())

    result = compute(expr, dfbig)

    groups = dfbig.groupby(['name', 'sex'])
    expected = DataFrame([['Alice', 'F', 204],
                          ['Drew', 'F', 104],
                          ['Drew', 'M', 310]], columns=['name', 'sex', '0'])
    expected.columns = expr.fields

    assert str(result) == str(expected)


def test_by_four():
    t = tbig[['sex', 'amount']]
    expr = by(t['sex'], t['amount'].max())
    result = compute(expr, dfbig)

    expected = DataFrame([['F', 100],
                          ['M', 200]], columns=['sex', 'amount_max'])

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

    t_idx = TableSymbol('t_idx', '{name: string, node_id: int32}')

    t_arc = TableSymbol('t_arc', '{node_out: int32, node_id: int32}')

    joined = join(t_arc, t_idx, "node_id")

    want = by(joined['name'], joined['node_id'].count())

    result = compute(want, {t_arc: df_arc, t_idx:df_idx})

    result_pandas = pd.merge(df_arc, df_idx, on='node_id')

    expected = result_pandas.groupby('name')['node_id'].count().reset_index()
    assert str(result.values) == str(expected.values)
    assert list(result.columns) == ['name', 'node_id_count']


def test_sort():
    print(str(compute(t.sort('amount'), df)))
    print(str(df.sort('amount')))
    assert str(compute(t.sort('amount'), df)) == str(df.sort('amount'))

    assert str(compute(t.sort('amount', ascending=True), df)) == \
            str(df.sort('amount', ascending=True))

    assert str(compute(t.sort(['amount', 'id']), df)) == \
            str(df.sort(['amount', 'id']))


def test_sort_on_series_no_warning(recwarn):
    expected = df.amount.order()

    recwarn.clear()

    assert str(compute(t['amount'].sort('amount'), df)) ==\
            str(expected)

    # raises as assertion error if no warning occurs, same thing for below
    with pytest.raises(AssertionError):
        assert recwarn.pop(FutureWarning)

    assert str(compute(t['amount'].sort(), df)) ==\
            str(expected)
    with pytest.raises(AssertionError):
        assert recwarn.pop(FutureWarning)


def test_head():
    assert str(compute(t.head(1), df)) == str(df.head(1))


def test_label():
    expected = df['amount'] * 10
    expected.name = 'foo'
    assert str(compute((t['amount'] * 10).label('foo'), df)) == \
            str(expected)


def test_relabel():
    result = compute(t.relabel({'name': 'NAME', 'id': 'ID'}), df)
    assert list(result.columns) == ['NAME', 'amount', 'ID']


def test_relabel_series():
    result = compute(t.relabel({'name': 'NAME'}), df.name)
    assert result.name == 'NAME'


@pytest.fixture
def tframe():
    ts = pd.date_range('now', periods=10).to_series().reset_index(drop=True)
    df = DataFrame({'timestamp': ts})
    return df


def test_map_with_rename(tframe):
    t = Table(tframe)
    result = t.timestamp.map(lambda x: x.date(), schema='{date: datetime}')
    renamed = result.relabel({'timestamp': 'date'})
    assert renamed.fields == ['date']


@pytest.mark.xfail(reason="Should this?  This seems odd but vacuously valid")
def test_multiple_renames_on_series_fails(tframe):
    t = Table(tframe)
    expr = t.timestamp.relabel({'timestamp': 'date', 'hello': 'world'})
    with pytest.raises(ValueError):
        compute(expr)


def test_map_column():
    inc = lambda x: x + 1
    result = compute(t['amount'].map(inc), df)
    expected = df['amount'] + 1
    assert str(result) == str(expected)


def test_map():
    f = lambda _, amt, id: amt + id
    result = compute(t.map(f), df)
    expected = df['amount'] + df['id']
    assert str(result) == str(expected)


def test_apply_column():
    result = compute(Apply(np.sum, t['amount']), df)
    expected = np.sum(df['amount'])

    assert str(result) == str(expected)

    result = compute(Apply(builtins.sum, t['amount']), df)
    expected = builtins.sum(df['amount'])

    assert str(result) == str(expected)


def test_apply():
    result = compute(Apply(str, t), df)
    expected = str(df)

    assert result == expected


def test_merge():
    col = (t['amount'] * 2).label('new')
    expr = merge(t['name'], col)

    expected = DataFrame([['Alice', 200],
                          ['Bob', 400],
                          ['Alice', 100]],
                         columns=['name', 'new'])

    result = compute(expr, df)

    assert str(result) == str(expected)


def test_by_nunique():
    result = compute(by(t['name'], t['id'].nunique()), df)
    expected = DataFrame([['Alice', 2], ['Bob', 1]],
                         columns=['name', 'id_nunique'])

    assert str(result) == str(expected)


def test_selection_out_of_order():
    expr = t['name'][t['amount'] < 100]

    assert str(compute(expr, df)) == str(df['name'][df['amount'] < 100])


def test_union():

    d1 = DataFrame([['Alice', 100, 1],
                    ['Bob', 200, 2],
                    ['Alice', 50, 3]], columns=['name', 'amount', 'id'])
    d2 = DataFrame([['Alice', 100, 4],
                    ['Bob', 200, 5],
                    ['Alice', 50, 6]], columns=['name', 'amount', 'id'])
    d3 = DataFrame([['Alice', 100, 7],
                    ['Bob', 200, 8],
                    ['Alice', 50, 9]], columns=['name', 'amount', 'id'])

    t1 = TableSymbol('t1', '{name: string, amount: int, id: int}')
    t2 = TableSymbol('t2', '{name: string, amount: int, id: int}')
    t3 = TableSymbol('t3', '{name: string, amount: int, id: int}')

    expr = union(t1, t2, t3)

    result = compute(expr, {t1: d1, t2: d2, t3: d3})

    assert np.all(result.columns == d1.columns)
    assert set(result['id']) == set(range(1, 10))


def test_outer_join():
    left = [(1, 'Alice', 100),
            (2, 'Bob', 200),
            (4, 'Dennis', 400)]
    left = DataFrame(left, columns=['id', 'name', 'amount'])

    right = [('NYC', 1),
             ('Boston', 1),
             ('LA', 3),
             ('Moscow', 4)]
    right = DataFrame(right, columns=['city', 'id'])

    L = TableSymbol('L', '{id: int, name: string, amount: real}')
    R = TableSymbol('R', '{city: string, id: int}')

    convert = lambda df: set(df.to_records(index=False).tolist())

    assert convert(compute(join(L, R), {L: left, R: right})) == set(
            [(1, 'Alice', 100, 'NYC'),
             (1, 'Alice', 100, 'Boston'),
             (4, 'Dennis', 400, 'Moscow')])

    assert convert(compute(join(L, R, how='left'), {L: left, R: right})) == set(
            [(1, 'Alice', 100, 'NYC'),
             (1, 'Alice', 100, 'Boston'),
             (2, 'Bob', 200, np.nan),
             (4, 'Dennis', 400, 'Moscow')])

    df = compute(join(L, R, how='right'), {L: left, R: right})
    expected = DataFrame(
            [(1., 'Alice', 100., 'NYC'),
             (1., 'Alice', 100., 'Boston'),
             (3., np.nan, np.nan, 'LA'),
             (4., 'Dennis', 400., 'Moscow')],
            columns=['id', 'name', 'amount', 'city'])

    assert str(df.sort('id').to_records(index=False)) ==\
            str(expected.sort('id').to_records(index=False))

    df = compute(join(L, R, how='outer'), {L: left, R: right})
    expected = DataFrame(
            [(1., 'Alice', 100., 'NYC'),
             (1., 'Alice', 100., 'Boston'),
             (2., 'Bob', 200., np.nan),
             (3., np.nan, np.nan, 'LA'),
             (4., 'Dennis', 400., 'Moscow')],
            columns=['id', 'name', 'amount', 'city'])

    assert str(df.sort('id').to_records(index=False)) ==\
            str(expected.sort('id').to_records(index=False))


def test_by_on_same_column():
    df = pd.DataFrame([[1,2],[1,4],[2,9]], columns=['id', 'value'])
    t = TableSymbol('data', dshape='{id:int, value:int}')

    gby = by(t['id'], t['id'].count())

    expected = DataFrame([[1, 2], [2, 1]], columns=['id', 'id_count'])
    result = compute(gby, {t:df})

    assert str(result) == str(expected)


def test_summary_by():
    expr = by(t.name, summary(count=t.id.count(), sum=t.amount.sum()))
    assert str(compute(expr, df)) == \
            str(DataFrame([['Alice', 2, 150],
                           ['Bob', 1, 200]], columns=['name', 'count', 'sum']))

    expr = by(t.name, summary(count=t.id.count(), sum=(t.amount + 1).sum()))
    assert str(compute(expr, df)) == \
            str(DataFrame([['Alice', 2, 152],
                           ['Bob', 1, 201]], columns=['name', 'count', 'sum']))


@pytest.mark.xfail(reason="reduction assumed to be at the end")
def test_summary_by_reduction_arithmetic():
    expr = by(t.name, summary(count=t.id.count(), sum=t.amount.sum() + 1))
    assert str(compute(expr, df)) == \
            str(DataFrame([['Alice', 2, 151],
                           ['Bob', 1, 202]], columns=['name', 'count', 'sum']))


def test_summary():
    expr = summary(count=t.id.count(), sum=t.amount.sum())
    assert str(compute(expr, df)) == str(Series({'count': 3, 'sum': 350}))


def test_dplyr_transform():
    df = DataFrame({'timestamp': pd.date_range('now', periods=5)})
    t = TableSymbol('t', discover(df))
    expr = transform(t, date=t.timestamp.map(lambda x: x.date(),
                                             schema='datetime'))
    lhs = compute(expr, df)
    rhs = pd.concat([df, Series(df.timestamp.map(lambda x: x.date()),
                                name='date').to_frame()], axis=1)
    assert str(lhs) == str(rhs)


def test_nested_transform():
    d = {'timestamp': [1379613528, 1379620047], 'platform': ["Linux",
                                                             "Windows"]}
    df = DataFrame(d)
    t = TableSymbol('t', discover(df))
    t = transform(t, timestamp=t.timestamp.map(datetime.fromtimestamp,
                                               schema='datetime'))
    expr = transform(t, date=t.timestamp.map(lambda x: x.date(),
                                             schema='datetime'))
    result = compute(expr, df)
    df['timestamp'] = df.timestamp.map(datetime.fromtimestamp)
    df['date'] = df.timestamp.map(lambda x: x.date())
    assert str(result) == str(df)


def test_like():
    expr = t.like(name='Alice*')
    expected = DataFrame([['Alice', 100, 1],
                          ['Alice', 50, 3]],
                         columns=['name', 'amount', 'id'])

    result = compute(expr, df).reset_index(drop=True)
    assert (result == expected).all().all()


def test_rowwise_by():
    f = lambda _, id, name: id + len(name)
    expr = by(t.map(f), t.amount.sum())

    df = pd.DataFrame({'id': [1, 1, 2],
                       'name': ['alice', 'wendy', 'bob'],
                       'amount': [100, 200, 300.03]})
    expected = pd.DataFrame([(5, 300.03), (6, 300)], columns=['index',
                                                              'amount_sum'])

    result = compute(expr, df)
    assert expected.index.tolist() == result.index.tolist()
    assert expected.columns.tolist() == result.columns.tolist()
    assert expected.values.tolist() == result.values.tolist()


def test_datetime_access():
    df = DataFrame({'name': ['Alice', 'Bob', 'Joe'],
                    'when': [datetime(2010, 1, 1, 1, 1, 1)] * 3,
                    'amount': [100, 200, 300],
                    'id': [1, 2, 3]})

    t = TableSymbol('t', discover(df))

    for attr in ['day', 'month', 'minute', 'second']:
        assert (compute(getattr(t.when, attr), df) == \
                Series([1, 1, 1])).all()


def test_slice():
    assert (compute(t[0], df) == df.iloc[0]).all()
    assert (compute(t[2], df) == df.iloc[2]).all()
    assert (compute(t[:2], df) == df.iloc[:2]).all().all()
    assert (compute(t[1:3], df) == df.iloc[1:3]).all().all()
    assert (compute(t[1::2], df) == df.iloc[1::2]).all().all()
