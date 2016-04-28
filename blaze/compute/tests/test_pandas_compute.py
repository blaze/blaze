from __future__ import absolute_import, division, print_function

import pytest

from datetime import datetime, timedelta

import numpy as np

import pandas as pd
import pandas.util.testing as tm
from pandas import DataFrame, Series

from string import ascii_lowercase

from blaze.compute.core import compute
from blaze.compute.pandas import pdsort
from blaze import dshape, discover, transform, broadcast_collect
from blaze.expr import symbol, join, by, summary, distinct, shape
from blaze.expr import (merge, exp, mean, count, nunique, sum, min, max, any,
                        var, std, concat, coalesce)
from blaze.compatibility import builtins, xfail, assert_series_equal


t = symbol('t', 'var * {name: string, amount: int, id: int}')
nt = symbol('t', 'var * {name: ?string, amount: float64, id: int}')


df = DataFrame([['Alice', 100, 1],
                ['Bob', 200, 2],
                ['Alice', 50, 3]], columns=['name', 'amount', 'id'])

ndf = DataFrame([['Alice', 100.0, 1],
                 ['Bob', np.nan, 2],
                 [np.nan, 50.0, 3]], columns=['name', 'amount', 'id'])


tbig = symbol('tbig',
              'var * {name: string, sex: string[1], amount: int, id: int}')

dfbig = DataFrame([['Alice', 'F', 100, 1],
                   ['Alice', 'F', 100, 3],
                   ['Drew', 'F', 100, 4],
                   ['Drew', 'M', 100, 5],
                   ['Drew', 'M', 200, 5]],
                  columns=['name', 'sex', 'amount', 'id'])


# for now jsut copy this, but will open a PR to see if we can remove some of
# the repetitive copying
tbgr = symbol('tbgr',
              """ var * {name: string,
                         sex: string[1],
                         amount: int,
                         id: int,
                         comment: ?string}
              """)

dfbgr = DataFrame([['Alice', 'F', 100, 1, 'Alice comment'],
                   ['Alice', 'F', 100, 3, None],
                   ['Drew', 'F', 100, 4, 'Drew comment'],
                   ['Drew', 'M', 100, 5, 'Drew comment 2'],
                   ['Drew', 'M', 200, 5, None]],
                  columns=['name', 'sex', 'amount', 'id', 'comment'])


@pytest.fixture(scope='module')
def df_add_null():
    rows = [(None, 'M', 300, 6),
            ('first', None, 300, 6),
            (None, None, 300, 6)]
    df_add_null = dfbig.append(DataFrame(rows,
                                         columns=dfbig.columns
                                         ), ignore_index=True)

    return df_add_null


def test_series_broadcast():
    s = Series([1, 2, 3], name='a')
    t = symbol('t', 'var * {a: int64}')
    bcast = broadcast_collect(expr=(2 * t.a - t.a**2))
    result = compute(bcast, s)
    assert_series_equal(result, 2 * s - s**2)


def test_frame_broadcast():
    bcast = broadcast_collect(expr=t.amount * t.id)
    result = compute(bcast, df)
    assert_series_equal(result, df.amount * df.id)


def test_series_columnwise():
    s = Series([1, 2, 3], name='a')
    t = symbol('t', 'var * {a: int64}')
    result = compute(t.a + 1, s)
    assert_series_equal(s + 1, result)


def test_symbol():
    tm.assert_frame_equal(compute(t, df), df)


def test_projection():
    tm.assert_frame_equal(compute(t[['name', 'id']], df),
                          df[['name', 'id']])


def test_eq():
    assert_series_equal(compute(t['amount'] == 100, df),
                           df['amount'] == 100)


def test_selection():
    tm.assert_frame_equal(compute(t[t['amount'] == 0], df),
                          df[df['amount'] == 0])
    tm.assert_frame_equal(compute(t[t['amount'] > 150], df),
                          df[df['amount'] > 150])


def test_arithmetic():
    assert_series_equal(compute(t['amount'] + t['id'], df),
                           df.amount + df.id)
    assert_series_equal(compute(t['amount'] * t['id'], df),
                           df.amount * df.id)
    assert_series_equal(compute(t['amount'] % t['id'], df),
                           df.amount % df.id)


def test_join():
    left = DataFrame(
        [['Alice', 100], ['Bob', 200]], columns=['name', 'amount'])
    right = DataFrame([['Alice', 1], ['Bob', 2]], columns=['name', 'id'])

    lsym = symbol('L', 'var * {name: string, amount: int}')
    rsym = symbol('R', 'var * {name: string, id: int}')
    joined = join(lsym, rsym, 'name')

    assert (dshape(joined.schema) ==
            dshape('{name: string, amount: int, id: int}'))

    result = compute(joined, {lsym: left, rsym: right})

    expected = DataFrame([['Alice', 100, 1], ['Bob', 200, 2]],
                         columns=['name', 'amount', 'id'])

    tm.assert_frame_equal(result, expected)
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

    lsym = symbol('lsym', 'var * {x: int, y: int, z: int}')
    rsym = symbol('rsym', 'var * {x: int, y: int, w: int}')

    j = join(lsym, rsym, ['x', 'y'])

    expected = [(1, 2, 3, 30),
                (1, 3, 5, 50),
                (1, 3, 5, 150)]
    expected = DataFrame(expected, columns=['x', 'y', 'z', 'w'])

    result = compute(j, {lsym: left, rsym: right})

    print(result)

    tm.assert_frame_equal(result, expected)
    assert list(result.columns) == list(j.fields)


def test_unary_op():
    assert (compute(exp(t['amount']), df) == np.exp(df['amount'])).all()


def test_abs():
    assert (compute(abs(t['amount']), df) == abs(df['amount'])).all()


def test_neg():
    assert_series_equal(compute(-t['amount'], df),
                           -df['amount'])


@xfail(reason='Projection does not support arithmetic')
def test_neg_projection():
    assert_series_equal(compute(-t[['amount', 'id']], df),
                           -df[['amount', 'id']])


def test_columns_series():
    assert isinstance(compute(t['amount'], df), Series)
    assert isinstance(compute(t['amount'] > 150, df), Series)


def test_reductions():
    assert compute(mean(t['amount']), df) == 350 / 3
    assert compute(count(t['amount']), df) == 3
    assert compute(sum(t['amount']), df) == 100 + 200 + 50
    assert compute(min(t['amount']), df) == 50
    assert compute(max(t['amount']), df) == 200
    assert compute(nunique(t['amount']), df) == 3
    assert compute(nunique(t['name']), df) == 2
    assert compute(any(t['amount'] > 150), df) is True
    assert compute(any(t['amount'] > 250), df) is False
    assert compute(var(t['amount']), df) == df.amount.var(ddof=0)
    assert compute(var(t['amount'], unbiased=True), df) == df.amount.var()
    assert compute(std(t['amount']), df) == df.amount.std(ddof=0)
    assert compute(std(t['amount'], unbiased=True), df) == df.amount.std()
    assert compute(t.amount[0], df) == df.amount.iloc[0]
    assert compute(t.amount[-1], df) == df.amount.iloc[-1]


def test_reductions_on_dataframes():
    assert compute(count(t), df) == 3
    assert shape(compute(count(t, keepdims=True), df)) == (1,)


def test_1d_reductions_keepdims():
    series = df['amount']
    for r in [sum, min, max, nunique, count, std, var]:
        result = compute(r(t.amount, keepdims=True), {t.amount: series})
        assert type(result) == type(series)


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
    d_t = distinct(tbig)
    d_df = compute(d_t, dftoobig)
    tm.assert_frame_equal(d_df, dfbig)
    # Test idempotence
    tm.assert_frame_equal(compute(d_t, d_df), d_df)


def test_distinct_on():
    cols = ['name', 'sex', 'amount', 'id']
    df = DataFrame([['Alice', 'F', 100, 1],
                    ['Alice', 'F', 100, 3],
                    ['Drew', 'F', 100, 4],
                    ['Drew', 'M', 100, 5],
                    ['Drew', 'F', 100, 4],
                    ['Drew', 'M', 100, 5],
                    ['Drew', 'M', 200, 5]],
                   columns=cols)
    s = symbol('s', discover(df))
    computed = compute(s.distinct('sex'), df)
    tm.assert_frame_equal(
        computed,
        pd.DataFrame([['Alice', 'F', 100, 1],
                      ['Drew', 'M', 100, 5]],
                     columns=cols),
    )


def test_by_one():
    result = compute(by(t['name'], total=t['amount'].sum()), df)
    expected = df.groupby('name')['amount'].sum().reset_index()
    expected.columns = ['name', 'total']
    tm.assert_frame_equal(result, expected)


def test_by_two():
    result = compute(by(tbig[['name', 'sex']],
                        total=sum(tbig['amount'])), dfbig)

    expected = DataFrame([['Alice', 'F', 200],
                          ['Drew', 'F', 100],
                          ['Drew', 'M', 300]],
                         columns=['name', 'sex', 'total'])

    tm.assert_frame_equal(result, expected)


def test_by_three():

    expr = by(tbig[['name', 'sex']],
              total=(tbig['id'] + tbig['amount']).sum())

    result = compute(expr, dfbig)

    expected = DataFrame([['Alice', 'F', 204],
                          ['Drew', 'F', 104],
                          ['Drew', 'M', 310]], columns=['name', 'sex', 'total'])
    expected.columns = expr.fields
    tm.assert_frame_equal(result, expected)


def test_by_four():
    t = tbig[['sex', 'amount']]
    expr = by(t['sex'], max=t['amount'].max())
    result = compute(expr, dfbig)

    expected = DataFrame([['F', 100],
                          ['M', 200]], columns=['sex', 'max'])

    tm.assert_frame_equal(result, expected)


def test_join_by_arcs():
    df_idx = DataFrame([['A', 1],
                        ['B', 2],
                        ['C', 3]],
                       columns=['name', 'node_id'])

    df_arc = DataFrame([[1, 3],
                        [2, 3],
                        [3, 1]],
                       columns=['node_out', 'node_id'])

    t_idx = symbol('t_idx', 'var * {name: string, node_id: int32}')

    t_arc = symbol('t_arc', 'var * {node_out: int32, node_id: int32}')

    joined = join(t_arc, t_idx, "node_id")

    want = by(joined['name'], count=joined['node_id'].count())

    result = compute(want, {t_arc: df_arc, t_idx: df_idx})

    result_pandas = pd.merge(df_arc, df_idx, on='node_id')

    gb = result_pandas.groupby('name')
    expected = gb.node_id.count().reset_index().rename(columns={
                                                       'node_id': 'count'
                                                       })

    tm.assert_frame_equal(result, expected)
    assert list(result.columns) == ['name', 'count']


def test_join_suffixes():
    df = pd.DataFrame(
        list(dict((k, n) for k in ascii_lowercase[:5]) for n in range(5)),
    )
    a = symbol('a', discover(df))
    b = symbol('b', discover(df))

    suffixes = '_x', '_y'
    joined = join(a, b, 'a', suffixes=suffixes)

    expected = pd.merge(df, df, on='a', suffixes=suffixes)
    result = compute(joined, {a: df, b: df})
    tm.assert_frame_equal(result, expected)


def test_join_promotion():
    a_data = pd.DataFrame([[0.0, 1.5], [1.0, 2.5]], columns=list('ab'))
    b_data = pd.DataFrame([[0, 1], [1, 2]], columns=list('ac'))
    a = symbol('a', discover(a_data))
    b = symbol('b', discover(b_data))

    joined = join(a, b, 'a')
    assert joined.dshape == dshape('var * {a: float64, b: float64, c: int64}')

    expected = pd.merge(a_data, b_data, on='a')
    result = compute(joined, {a: a_data, b: b_data})
    tm.assert_frame_equal(result, expected)


def test_sort():
    tm.assert_frame_equal(compute(t.sort('amount'), df),
                          pdsort(df, 'amount'))

    tm.assert_frame_equal(compute(t.sort('amount', ascending=True), df),
                          pdsort(df, 'amount', ascending=True))

    tm.assert_frame_equal(compute(t.sort(['amount', 'id']), df),
                          pdsort(df, ['amount', 'id']))


def test_sort_on_series_no_warning(recwarn):
    expected = df.amount.order()

    recwarn.clear()

    assert_series_equal(compute(t['amount'].sort('amount'), df), expected)

    # raises as assertion error if no warning occurs, same thing for below
    with pytest.raises(AssertionError):
        assert recwarn.pop(FutureWarning)

    assert_series_equal(compute(t['amount'].sort(), df), expected)
    with pytest.raises(AssertionError):
        assert recwarn.pop(FutureWarning)


def test_field_on_series():
    expr = symbol('s', 'var * int')
    data = Series([1, 2, 3, 4], name='s')
    assert_series_equal(compute(expr.s, data), data)


def test_head():
    tm.assert_frame_equal(compute(t.head(1), df), df.head(1))


def test_tail():
    tm.assert_frame_equal(compute(t.tail(1), df), df.tail(1))


def test_sample():
    samp = compute(t.sample(n=2), df)
    assert len(samp) == 2


def test_sample_frac_rounding_edge_case():
    samp_big = compute(tbig.sample(frac=0.1), dfbig)
    assert len(samp_big) == int(np.ceil(len(dfbig) * 0.1))


def test_sample_clip():
    samp_series = compute(t.name.sample(n=2*len(df)), df)
    samp_df = compute(t.sample(n=2*len(df)), df)
    assert len(samp_series) == len(samp_df) == len(df)


def test_label():
    expected = df['amount'] * 10
    expected.name = 'foo'
    assert_series_equal(compute((t['amount'] * 10).label('foo'), df),
                           expected)


def test_relabel():
    result = compute(t.relabel({'name': 'NAME', 'id': 'ID'}), df)
    expected = df.rename(columns={'name': 'NAME', 'id': 'ID'})
    tm.assert_frame_equal(result, expected)


def test_relabel_series():
    result = compute(t.relabel({'name': 'NAME'}), df.name)
    assert result.name == 'NAME'


ts = pd.date_range('now', periods=10).to_series().reset_index(drop=True)
tframe = DataFrame({'timestamp': ts})


def test_map_column():
    inc = lambda x: x + 1
    result = compute(t['amount'].map(inc, 'int'), df)
    expected = df['amount'] + 1
    assert_series_equal(result, expected)


def test_map():
    f = lambda _, amt, id: amt + id
    result = compute(t.map(f, 'real'), df)
    expected = df['amount'] + df['id']
    assert_series_equal(result, expected)


def test_apply_column():
    result = compute(t.amount.apply(np.sum, 'real'), df)
    expected = np.sum(df['amount'])
    assert result == expected

    result = compute(t.amount.apply(builtins.sum, 'real'), df)
    expected = builtins.sum(df['amount'])
    assert result == expected


def test_apply():
    result = compute(t.apply(str, 'string'), df)
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
    tm.assert_frame_equal(result, expected)


def test_by_nunique():
    result = compute(by(t['name'], count=t['id'].nunique()), df)
    expected = DataFrame([['Alice', 2], ['Bob', 1]],
                         columns=['name', 'count'])

    tm.assert_frame_equal(result, expected)


def test_selection_out_of_order():
    expr = t['name'][t['amount'] < 100]
    expected = df.loc[df.amount < 100, 'name']
    result = compute(expr, df)
    assert_series_equal(result, expected)


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

    lsym = symbol('lsym', 'var * {id: int, name: string, amount: real}')
    rsym = symbol('rsym', 'var * {city: string, id: int}')

    convert = lambda df: set(df.to_records(index=False).tolist())

    assert (convert(compute(join(lsym, rsym), {lsym: left, rsym: right})) ==
            set([(1, 'Alice', 100, 'NYC'),
                 (1, 'Alice', 100, 'Boston'),
                 (4, 'Dennis', 400, 'Moscow')]))

    assert (convert(compute(join(lsym, rsym, how='left'),
                            {lsym: left, rsym: right})) ==
            set([(1, 'Alice', 100, 'NYC'),
                 (1, 'Alice', 100, 'Boston'),
                 (2, 'Bob', 200, np.nan),
                 (4, 'Dennis', 400, 'Moscow')]))

    df = compute(join(lsym, rsym, how='right'), {lsym: left, rsym: right})
    expected = DataFrame([(1., 'Alice', 100., 'NYC'),
                          (1., 'Alice', 100., 'Boston'),
                          (3., np.nan, np.nan, 'lsymA'),
                          (4., 'Dennis', 400., 'Moscow')],
                         columns=['id', 'name', 'amount', 'city'])

    result = pdsort(df, 'id').to_records(index=False)
    expected = pdsort(expected, 'id').to_records(index=False)
    np.array_equal(result, expected)

    df = compute(join(lsym, rsym, how='outer'), {lsym: left, rsym: right})
    expected = DataFrame([(1., 'Alice', 100., 'NYC'),
                          (1., 'Alice', 100., 'Boston'),
                          (2., 'Bob', 200., np.nan),
                          (3., np.nan, np.nan, 'LA'),
                          (4., 'Dennis', 400., 'Moscow')],
                         columns=['id', 'name', 'amount', 'city'])

    result = pdsort(df, 'id').to_records(index=False)
    expected = pdsort(expected, 'id').to_records(index=False)
    np.array_equal(result, expected)


def test_by_on_same_column():
    df = pd.DataFrame([[1, 2], [1, 4], [2, 9]], columns=['id', 'value'])
    t = symbol('data', 'var * {id: int, value: int}')

    gby = by(t['id'], count=t['id'].count())

    expected = DataFrame([[1, 2], [2, 1]], columns=['id', 'count'])
    result = compute(gby, {t: df})
    tm.assert_frame_equal(result, expected)


def test_summary_by():
    expr = by(t.name, summary(count=t.id.count(), sum=t.amount.sum()))
    result = compute(expr, df)
    expected = DataFrame([['Alice', 2, 150],
                          ['Bob', 1, 200]], columns=['name', 'count', 'sum'])

    expr = by(t.name, summary(count=t.id.count(), sum=(t.amount + 1).sum()))
    result = compute(expr, df)
    expected = DataFrame([['Alice', 2, 152],
                          ['Bob', 1, 201]], columns=['name', 'count', 'sum'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.xfail(raises=TypeError,
                   reason=('pandas backend cannot support non Reduction '
                           'subclasses'))
def test_summary_by_first():
    expr = by(t.name, fst=t.amount[0])
    result = compute(expr, df)
    assert result == df.amount.iloc[0]


def test_summary_by_reduction_arithmetic():
    expr = by(t.name, summary(count=t.id.count(), sum=t.amount.sum() + 1))
    result = compute(expr, df)
    expected = DataFrame([['Alice', 2, 151],
                          ['Bob', 1, 201]], columns=['name', 'count', 'sum'])
    tm.assert_frame_equal(result, expected)


def test_summary():
    expr = summary(count=t.id.count(), sum=t.amount.sum())
    assert_series_equal(compute(expr, df), Series({'count': 3, 'sum': 350}))


def test_summary_on_series():
    ser = Series([1, 2, 3])
    s = symbol('s', '3 * int')
    expr = summary(max=s.max(), min=s.min())
    assert compute(expr, ser) == (3, 1)

    expr = summary(max=s.max(), min=s.min(), keepdims=True)
    assert compute(expr, ser) == [(3, 1)]


def test_summary_keepdims():
    expr = summary(count=t.id.count(), sum=t.amount.sum(), keepdims=True)
    expected = DataFrame([[3, 350]], columns=['count', 'sum'])
    tm.assert_frame_equal(compute(expr, df), expected)


def test_dplyr_transform():
    df = DataFrame({'timestamp': pd.date_range('now', periods=5)})
    t = symbol('t', discover(df))
    expr = transform(t, date=t.timestamp.map(lambda x: x.date(),
                                             schema='datetime'))
    lhs = compute(expr, df)
    rhs = pd.concat([df, Series(df.timestamp.map(lambda x: x.date()),
                                name='date').to_frame()], axis=1)
    tm.assert_frame_equal(lhs, rhs)


def test_nested_transform():
    d = {'timestamp': [1379613528, 1379620047], 'platform': ["Linux",
                                                             "Windows"]}
    df = DataFrame(d)
    t = symbol('t', discover(df))
    t = transform(t, timestamp=t.timestamp.map(datetime.fromtimestamp,
                                               schema='datetime'))
    expr = transform(t, date=t.timestamp.map(lambda x: x.date(),
                                             schema='datetime'))
    result = compute(expr, df)
    df['timestamp'] = df.timestamp.map(datetime.fromtimestamp)
    df['date'] = df.timestamp.map(lambda x: x.date())
    tm.assert_frame_equal(result, df)


def test_transform_with_common_subexpression():
    df = DataFrame(np.random.rand(5, 2), columns=list('ab'))
    t = symbol('t', discover(df))
    expr = transform(t, c=t.a - t.a % 3, d=t.a % 3)
    result = compute(expr, df)
    expected = pd.concat(
        [df[c] for c in df.columns] + [
            pd.Series(df.a - df.a % 3, name='c'),
            pd.Series(df.a % 3, name='d')
        ],
        axis=1
    )
    tm.assert_frame_equal(result, expected)


def test_merge_with_common_subexpression():
    df = DataFrame(np.random.rand(5, 2), columns=list('ab'))
    t = symbol('t', discover(df))
    expr = merge((t.a - t.a % 3).label('a'), (t.a % 3).label('b'))
    result = compute(expr, {t: df})
    expected = pd.concat(
        [
            pd.Series(df.a - df.a % 3, name='a'),
            pd.Series(df.a % 3, name='b')
        ],
        axis=1
    )
    tm.assert_frame_equal(result, expected)


def test_like():
    expr = t[t.name.like('Alice*')]
    expected = DataFrame([['Alice', 100, 1],
                          ['Alice', 50, 3]],
                         columns=['name', 'amount', 'id'])

    result = compute(expr, df).reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


def test_str_len():
    expr = t.name.str_len()
    expected = pd.Series([5, 3, 5], name='name')
    result = compute(expr, df).reset_index(drop=True)
    assert_series_equal(expected, result)


def test_str_upper():
    expr = t.name.str_upper()
    expected = pd.Series(['ALICE', 'BOB', 'ALICE'], name='name')
    result = compute(expr, df).reset_index(drop=True)
    assert_series_equal(expected, result)


def test_str_lower():
    expr = t.name.str_lower()
    expected = pd.Series(['alice', 'bob', 'alice'], name='name')
    result = compute(expr, df).reset_index(drop=True)
    assert_series_equal(expected, result)


def test_str_cat():
    res = compute(tbig.name.str_cat(tbig.sex), dfbig)
    assert all(dfbig.name.str.cat(dfbig.sex) == res)


def test_str_cat_sep():
    res = compute(tbig.name.str_cat(tbig.sex, sep=' -- '), dfbig)
    assert all(dfbig.name.str.cat(dfbig.sex, sep=' -- ') == res)


def test_str_cat_null_row(df_add_null):
    res = compute(tbig.name.str_cat(tbig.sex, sep=' -- '), df_add_null)
    exp_res = df_add_null.name.str.cat(df_add_null.sex, sep=' -- ')

    assert all(exp_res.isnull() == res.isnull())
    assert all(exp_res[~exp_res.isnull()] == res[~res.isnull()])


def test_str_cat_chain_operation():
    expr = tbgr.name.str_cat(tbgr.comment.str_cat(tbgr.sex, sep=' --- '),
                             sep=' +++ ')
    res = compute(expr, dfbgr)
    exp_res = dfbgr.name.str.cat(dfbgr.comment.str.cat(dfbgr.sex, sep=' --- '),
                                 sep=' +++ ')
    assert all(exp_res.isnull() == res.isnull())
    assert all(exp_res[~exp_res.isnull()] == res[~res.isnull()])


def test_rowwise_by():
    f = lambda _, id, name: id + len(name)
    expr = by(t.map(f, 'int'), total=t.amount.sum())

    df = pd.DataFrame({'id': [1, 1, 2],
                       'name': ['alice', 'wendy', 'bob'],
                       'amount': [100, 200, 300.03]})
    expected = pd.DataFrame([(5, 300.03), (6, 300)], columns=expr.fields)

    result = compute(expr, df)
    tm.assert_frame_equal(result, expected)


def test_datetime_access():
    df = DataFrame({'name': ['Alice', 'Bob', 'Joe'],
                    'when': [datetime(2010, 1, 1, 1, 1, 1)] * 3,
                    'amount': [100, 200, 300],
                    'id': [1, 2, 3]})

    t = symbol('t', discover(df))

    for attr in ['day', 'month', 'minute', 'second']:
        expr = getattr(t.when, attr)
        assert_series_equal(compute(expr, df),
                            Series([1, 1, 1], name=expr._name))


def test_frame_slice():
    assert_series_equal(compute(t[0], df), df.iloc[0])
    assert_series_equal(compute(t[2], df), df.iloc[2])
    tm.assert_frame_equal(compute(t[:2], df), df.iloc[:2])
    tm.assert_frame_equal(compute(t[1:3], df), df.iloc[1:3])
    tm.assert_frame_equal(compute(t[1::2], df), df.iloc[1::2])
    tm.assert_frame_equal(compute(t[[2, 0]], df), df.iloc[[2, 0]])


def test_series_slice():
    assert compute(t.amount[0], df) == df.amount.iloc[0]
    assert compute(t.amount[2], df) == df.amount.iloc[2]
    assert_series_equal(compute(t.amount[:2], df), df.amount.iloc[:2])
    assert_series_equal(compute(t.amount[1:3], df), df.amount.iloc[1:3])
    assert_series_equal(compute(t.amount[1::2], df), df.amount.iloc[1::2])


def test_nelements():
    assert compute(t.nelements(), df) == len(df)
    assert compute(t.nrows, df) == len(df)


def test_datetime_truncation_minutes():
    data = Series(['2000-01-01T12:10:00Z', '2000-06-25T12:35:12Z'],
                  dtype='M8[ns]')
    s = symbol('s', 'var * datetime')
    result = compute(s.truncate(20, 'minutes'), data)
    expected = Series(['2000-01-01T12:00:00Z', '2000-06-25T12:20:00Z'],
                      dtype='M8[ns]', name='s')
    assert_series_equal(result, expected)


def test_datetime_truncation_nanoseconds():
    data = Series(['2000-01-01T12:10:00.000000005',
                   '2000-01-01T12:10:00.000000025'],
                  dtype='M8[ns]')
    s = symbol('s', 'var * datetime')
    expected = Series(['2000-01-01T12:10:00.000000000',
                       '2000-01-01T12:10:00.000000020'],
                      dtype='M8[ns]', name='s')
    result = compute(s.truncate(nanoseconds=20), data)
    assert_series_equal(result, expected)


def test_datetime_truncation_weeks():
    data = Series(['2000-01-01T12:10:00Z', '2000-06-25T12:35:12Z'],
                  dtype='M8[ns]')
    s = symbol('s', 'var * datetime')
    result = compute(s.truncate(2, 'weeks'), data)
    expected = Series(['1999-12-19', '2000-06-18'], dtype='M8[ns]', name='s')
    assert_series_equal(result, expected)


def test_datetime_truncation_days():
    data = Series(['2000-01-01T12:10:00Z', '2000-06-25T12:35:12Z'],
                  dtype='M8[ns]')
    s = symbol('s', 'var * datetime')
    result = compute(s.truncate(days=3), data)
    expected = Series(['1999-12-31', '2000-06-25'], dtype='M8[ns]', name='s')
    assert_series_equal(result, expected)


def test_datetime_truncation_same_as_python():
    data = Series(['2000-01-01T12:10:00Z', '2000-06-25T12:35:12Z'],
                  dtype='M8[ns]')
    s = symbol('s', 'var * datetime')
    assert (compute(s.truncate(weeks=2), data[0].to_pydatetime()) ==
            datetime(1999, 12, 26).date())


def test_complex_group_by():
    expr = by(merge(tbig.amount // 10, tbig.id % 2),
              count=tbig.name.count())
    result = compute(expr, dfbig)  # can we do this? yes we can!
    expected = dfbig.groupby([dfbig.amount // 10,
                              dfbig.id % 2])['name'].count().reset_index()
    expected = expected.rename(columns={'name': 'count'})
    tm.assert_frame_equal(result, expected)


def test_by_with_complex_summary():
    expr = by(t.name, total=t.amount.sum() + t.id.sum() - 1, a=t.id.min())
    result = compute(expr, df)
    assert list(result.columns) == expr.fields
    assert list(result.total) == [150 + 4 - 1, 200 + 2 - 1]


def test_notnull():
    assert (compute(nt.name.notnull(), ndf) == ndf.name.notnull()).all()


def test_isnan():
    assert (compute(nt.amount.isnan(), ndf) == ndf.amount.isnull()).all()


@pytest.mark.parametrize('keys', [[1], [2, 3]])
def test_isin(keys):
    expr = t[t.id.isin(keys)]
    result = compute(expr, df)
    expected = df.loc[df.id.isin(keys)]
    tm.assert_frame_equal(result, expected)


def test_nunique_table():
    expr = t.nunique()
    result = compute(expr, df)
    assert result == len(df.drop_duplicates())


def test_str_concat():
    a = Series(('a', 'b', 'c'))
    s = symbol('s', "3 * string[1, 'U32']")
    expr = s + 'a'
    assert (compute(expr, a) == (a + 'a')).all()


def test_str_repeat():
    a = Series(('a', 'b', 'c'))
    s = symbol('s', "3 * string[1, 'U32']")
    expr = s.repeat(3)
    assert (compute(expr, a) == (a * 3)).all()


def test_str_interp():
    a = Series(('%s', '%s', '%s'))
    s = symbol('s', "3 * string[1, 'U32']")
    expr = s.interp(1)
    assert (compute(expr, a) == (a % 1)).all()


def test_timedelta_arith():
    series = Series(pd.date_range('2014-01-01', '2014-02-01'))
    sym = symbol('s', discover(series))
    delta = timedelta(days=1)
    assert (compute(sym + delta, series) == series + delta).all()
    assert (compute(sym - delta, series) == series - delta).all()
    assert (
        compute(sym - (sym - delta), series) ==
        series - (series - delta)
    ).all()


@pytest.mark.parametrize('func,expected', (
    ('var', timedelta(0, 8, 250000)),
    ('std', timedelta(0, 2, 872281)),
))
def test_timedelta_stat_reduction(func, expected):
    deltas = pd.Series([timedelta(seconds=n) for n in range(10)])
    sym = symbol('s', discover(deltas))
    assert compute(getattr(sym, func)(), deltas) == expected


def test_coerce_series():
    s = pd.Series(list('123'), name='a')
    t = symbol('t', discover(s))
    result = compute(t.coerce(to='int64'), s)
    expected = pd.Series([1, 2, 3], name=s.name)
    assert_series_equal(result, expected)


def test_concat_arr():
    s_data = Series(np.arange(15))
    t_data = Series(np.arange(15, 30))

    s = symbol('s', discover(s_data))
    t = symbol('t', discover(t_data))

    assert (
        compute(concat(s, t), {s: s_data, t: t_data}) == Series(np.arange(30))
    ).all()


def test_concat_mat():
    s_data = DataFrame(np.arange(15).reshape(5, 3), columns=list('abc'))
    t_data = DataFrame(np.arange(15, 30).reshape(5, 3), columns=list('abc'))

    s = symbol('s', discover(s_data))
    t = symbol('t', discover(t_data))

    tm.assert_frame_equal(
        compute(concat(s, t), {s: s_data, t: t_data}),
        pd.DataFrame(np.arange(30).reshape(10, 3), columns=list('abc')),
    )


def test_count_keepdims_frame():
    df = pd.DataFrame(dict(a=[1, 2, 3, np.nan]))
    s = symbol('s', discover(df))
    assert_series_equal(compute(s.count(keepdims=True), df),
                        pd.Series([df.shape[0]], name='s_count'))


def test_time_field():
    data = pd.Series(pd.date_range(start='20120101', end='20120102', freq='H'))
    s = symbol('s', discover(data))
    result = compute(s.time, data)
    expected = data.dt.time
    expected.name = 's_time'
    assert_series_equal(result, expected)



@pytest.mark.parametrize('n', [-1, 0, 1])
def test_shift(n):
    data = pd.Series(pd.date_range(start='20120101', end='20120102', freq='H'))
    s = symbol('s', discover(data))
    result = compute(s.shift(n), data)
    expected = data.shift(n)
    assert_series_equal(result, expected)


def test_selection_inner_inputs():
    s_data = pd.DataFrame({'a': np.arange(5)})
    t_data = pd.DataFrame({'a': np.arange(5)})

    s = symbol('s', 'var * {a: int64}')
    t = symbol('t', 'var * {a: int64}')

    tm.assert_frame_equal(
        compute(s[s.a == t.a], {s: s_data, t: t_data}),
        s_data
    )


def test_by_with_reduction_on_df():
    expr = by(tbig.name, id_sum=tbig.id.sum(), count=tbig.count())
    compute(expr, dfbig)


def test_coalesce():
    data = pd.Series([0, None, 1, None, 2, None], dtype=object)

    s = symbol('s', 'var * ?int')
    t = symbol('t', 'int')
    u = symbol('u', '?int')
    v = symbol('v', 'var * int')
    w = symbol('w', 'var * ?int')

    # array to scalar
    tm.assert_series_equal(
        compute(coalesce(s, t), {s: data, t: -1}),
        pd.Series([0, -1, 1, -1, 2, -1], dtype=object),
    )
    # array to scalar with NULL
    tm.assert_series_equal(
        compute(coalesce(s, u), {s: data, u: None}),
        pd.Series([0, None, 1, None, 2, None], dtype=object),
    )
    # array to array
    tm.assert_series_equal(
        compute(coalesce(s, v), {
            s: data, v: np.array([-1, -2, -3, -4, -5, -6]),
        }),
        pd.Series([0, -2, 1, -4, 2, -6], dtype=object),
    )
    # array to array with NULL
    tm.assert_series_equal(
        compute(coalesce(s, w), {
            s: data, w: np.array([-1, None, -3, -4, -5, -6]),
        }),
        pd.Series([0, None, 1, -4, 2, -6], dtype=object),
    )
