import pytest

import numpy as np
import pandas as pd
import pandas.util.testing as tm

dd = pytest.importorskip('dask.dataframe')

from blaze import symbol, discover, compute, broadcast_collect
from blaze.expr import mean, count, sum, min, max, var, std, summary


def eq(a, b):
    if isinstance(a, dd.DataFrame):
        tm.assert_frame_equal(a.compute(), b)
    elif isinstance(b, dd.Series):
        tm.assert_series_equal(a.compute(), b)


t = symbol('t', 'var * {name: string, amount: int, id: int}')
nt = symbol('t', 'var * {name: ?string, amount: float64, id: int}')

df = pd.DataFrame([['Alice', 100, 1],
                   ['Bob', 200, 2],
                   ['Alice', 50, 3]], columns=['name', 'amount', 'id'])

ndf = pd.DataFrame([['Alice', 100.0, 1],
                    ['Bob', np.nan, 2],
                    [np.nan, 50.0, 3]], columns=['name', 'amount', 'id'])

tbig = symbol('tbig',
              'var * {name: string, sex: string[1], amount: int, id: int}')

dfbig = pd.DataFrame([['Alice', 'F', 100, 1],
                      ['Alice', 'F', 100, 3],
                      ['Drew', 'F', 100, 4],
                      ['Drew', 'M', 100, 5],
                      ['Drew', 'M', 200, 5]],
                     columns=['name', 'sex', 'amount', 'id'])

ddf = dd.from_pandas(df, npartitions=2)
ddfbig = dd.from_pandas(dfbig, npartitions=2)


def test_broadcast():
    bcast = broadcast_collect(expr=t.amount * t.id)
    result = compute(bcast, ddf)
    eq(result, df.amount * df.id)


def test_symbol():
    eq(compute(t, ddf), df)


def test_sample():
    assert len(compute(t.sample(n=2), ddf)) == len(df.sample(n=2))
    assert len(compute(t.sample(frac=0.4), ddf)) == len(df.sample(frac=0.4))


def test_head():
    eq(compute(t.head(1), ddf), df.head(1))


def test_tail():
    eq(compute(t.tail(1), ddf), df.tail(1))


def test_relabel():
    result = compute(t.relabel({'name': 'NAME', 'id': 'ID'}), ddf)
    expected = df.rename(columns={'name': 'NAME', 'id': 'ID'})
    eq(result, expected)


def test_relabel_series():
    result = compute(t.relabel({'name': 'NAME'}), ddf.name)
    assert result.name == 'NAME'


def test_series_columnwise():
    data = pd.Series([1, 2, 3, 4], name='a')
    s = dd.from_pandas(data, npartitions=2)
    t = symbol('t', 'var * {a: int64}')
    result = compute(t.a + 1, s)
    eq(result, data)


def test_projection():
    eq(compute(t[['name', 'id']], ddf), df[['name', 'id']])


def test_field_on_series():
    expr = symbol('s', 'var * int')
    data = pd.Series([1, 2, 3, 4], name='s')
    s = dd.from_pandas(data, npartitions=2)
    eq(compute(expr.s, s), data)


def test_selection():
    eq(compute(t[t['amount'] == 0], ddf), df[df['amount'] == 0])
    eq(compute(t[t['amount'] > 150], ddf), df[df['amount'] > 150])


def test_selection_out_of_order():
    expr = t['name'][t['amount'] < 100]
    expected = df.loc[df.amount < 100, 'name']
    result = compute(expr, ddf)
    eq(result, expected)


def test_selection_inner_inputs():
    s_data = pd.DataFrame({'a': np.arange(5)})
    t_data = pd.DataFrame({'a': np.arange(5)})

    s_dd = dd.from_pandas(s_data, npartitions=2)
    t_dd = dd.from_pandas(t_data, npartitions=2)

    s = symbol('s', 'var * {a: int64}')
    t = symbol('t', 'var * {a: int64}')

    eq(compute(s[s.a == t.a], {s: s_dd, t: t_dd}), s_data)


def test_reductions():
    assert compute(mean(t['amount']), ddf) == 350 / 3
    assert compute(count(t['amount']), ddf) == 3
    assert compute(sum(t['amount']), ddf) == 100 + 200 + 50
    assert compute(min(t['amount']), ddf) == 50
    assert compute(max(t['amount']), ddf) == 200
    assert compute(var(t['amount']), ddf) == df.amount.var(ddof=0)
    assert compute(var(t['amount'], unbiased=True), ddf) == df.amount.var()
    assert compute(std(t['amount']), ddf) == df.amount.std(ddof=0)
    assert compute(std(t['amount'], unbiased=True), ddf) == df.amount.std()


def test_summary():
    expr = summary(count=t.id.count(), sum=t.amount.sum())
    eq(compute(expr, ddf), pd.Series({'count': 3, 'sum': 350}))


def test_summary_on_series():
    ser = dd.from_pandas(pd.Series([1, 2, 3]), npartitions=2)
    s = symbol('s', '3 * int')
    expr = summary(max=s.max(), min=s.min())
    assert compute(expr, ser) == (3, 1)

    expr = summary(max=s.max(), min=s.min(), keepdims=True)
    assert compute(expr, ser) == [(3, 1)]


@pytest.mark.parametrize('keys', [[1], [2, 3]])
def test_isin(keys):
    expr = t[t.id.isin(keys)]
    result = compute(expr, ddf)
    expected = df.loc[df.id.isin(keys)]
    eq(result, expected)


def test_coerce_series():
    s = pd.Series(list('1234'), name='a')
    dds = dd.from_pandas(s, npartitions=2)
    t = symbol('t', discover(s))
    result = compute(t.coerce(to='int64'), dds)
    expected = pd.Series([1, 2, 3, 4], name=s.name)
    eq(result, expected)


def test_nelements_count():
    assert compute(t.nelements(), ddf) == len(df)
    assert compute(t.nrows, ddf) == len(df)
    assert compute(count(t), ddf) == len(df)
