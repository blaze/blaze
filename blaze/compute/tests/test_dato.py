import pytest

pytest.importorskip('graphlab')

from itertools import product
import operator as opr
import numpy as np
import graphlab as gl
import pandas as pd
import pandas.util.testing as tm
from odo import odo
from graphlab import aggregate as agg
from blaze import by, compute, symbol, discover, join, transform


@pytest.fixture
def nested():
    return gl.SFrame({'a': [{'b': 1}, {'a': 2, 'b': 3}, {'b': 3}, {'b': 4},
                            {'b': 5}, {'c': 6, 'b': 3}],
                      'b': list('abcab') + [None]})


@pytest.fixture
def nt(nested):
    return symbol('nt', discover(nested))


@pytest.fixture
def t(tf):
    return symbol('t', discover(tf))


@pytest.fixture
def tf():
    return gl.SFrame({'a': [1, 2, 3, 4, 5, 6],
                      'b': list('abcabc'),
                      'c': np.random.randn(6)})


@pytest.fixture
def s(sf):
    return symbol('s', discover(sf))


@pytest.fixture
def sf():
    return gl.SFrame({'a': [1, 2, 3, 3, 1, 1, 2, 3, 4, 4, 3, 1],
                      'b': list('bacbaccaacbb'),
                      'c': np.random.rand(12)})


@pytest.fixture
def d(df):
    return symbol('d', discover(df))


@pytest.fixture
def df(sf):
    dates = [x.to_pydatetime() for x in pd.date_range(start='20140101',
                                                      freq='S',
                                                      periods=len(sf))]
    sf['dates'] = gl.SArray(dates)
    return sf


@pytest.fixture
def dfu(sf):
    dates = [x.to_pydatetime() for x in pd.date_range(start='20140101',
                                                      freq='U',
                                                      periods=len(sf))]
    sf['dates'] = gl.SArray(dates)
    return sf


def test_projection(t, tf):
    expr = t[['a', 'c']]
    result = compute(expr, tf)
    expected = compute(expr, odo(tf, pd.DataFrame))
    tm.assert_frame_equal(odo(result, pd.DataFrame), expected)


@pytest.mark.parametrize('field', list('abc'))
def test_field(t, tf, field):
    expr = getattr(t, field)
    result = compute(expr, tf)
    expected = tf[field]
    tm.assert_series_equal(odo(result, pd.Series), odo(expected, pd.Series))


@pytest.mark.parametrize(['field', 'agg'],
                         product(list('ac'), ['mean', 'sum', 'max', 'min']))
def test_reduction(t, tf, field, agg):
    expr = getattr(t[field], agg)()
    result = compute(expr, tf)
    expected = getattr(tf[field], agg)()
    assert result == expected


@pytest.mark.parametrize(['field', 'ascending'],
                         product(['a', ['b'], ['a', 'c'], ['b', 'c']],
                                 [True, False]))
def test_sort_tframe(t, tf, field, ascending):
    expr = t.sort(field, ascending=ascending)
    result = compute(expr, tf)
    expected = tf.sort(sort_columns=field, ascending=ascending)
    tm.assert_frame_equal(odo(result, pd.DataFrame),
                          odo(expected, pd.DataFrame))


@pytest.mark.parametrize('op',
                         [pytest.mark.xfail(opr.truediv, raises=TypeError),
                          pytest.mark.xfail(opr.pow, raises=TypeError),
                          opr.sub])
def test_arith(t, tf, op):
    expr = op(t.a * 2 + t.c, 2 + 3)
    result = compute(expr, tf)
    expected = op(tf['a'] * 2 + tf['c'], 2 + 3)
    tm.assert_series_equal(odo(result, pd.Series), odo(expected, pd.Series))


def test_by(t, tf):
    expr = by(t.b, a_sum=t.a.sum())
    result = compute(expr, tf)
    expected = tf.groupby('b', operations={'a_sum': agg.SUM('a')})
    tm.assert_frame_equal(odo(result, pd.DataFrame),
                          odo(expected, pd.DataFrame))


@pytest.mark.parametrize(['how', 'field'],
                         product(['inner', 'outer', 'left', 'right'],
                                 ['a', list('ab')]))
def test_join(t, s, tf, sf, how, field):
    expr = join(t, s, field, how=how)
    result = compute(expr, {t: tf, s: sf})
    expected = tf.join(sf, on=field, how=how)[tf.column_names()]
    tm.assert_frame_equal(odo(result, pd.DataFrame),
                          odo(expected, pd.DataFrame))


def test_selection_table(t, tf):
    expr = t[t.c > 0.5]
    result = compute(expr, tf)
    expected = tf[tf['c'] > 0.5]
    tm.assert_frame_equal(odo(result, pd.DataFrame),
                          odo(expected, pd.DataFrame))


def test_selection_column(t, tf):
    expr = t.c[t.c > 0.5]
    result = compute(expr, tf)
    expected = tf['c'][tf['c'] > 0.5]
    tm.assert_series_equal(odo(result, pd.Series), odo(expected, pd.Series))


def test_nested_sframe(nt, nested):
    expr = nt.a.b
    result = compute(expr, nested)
    expected = nested['a'].unpack('', limit=['b'])['b']
    tm.assert_series_equal(odo(result, pd.Series), odo(expected, pd.Series))


def test_groupby_on_nested(nt, nested):
    expr = by(nt.a.b, avg=nt.a.c.mean())
    result = compute(expr, nested)
    unpacked = nested['a'].unpack('', limit=list('bc'))
    expected = unpacked.groupby('b', operations={'avg': agg.MEAN('c')})
    tm.assert_frame_equal(odo(result, pd.DataFrame),
                          odo(expected, pd.DataFrame))


def test_nelements(t, tf):
    assert compute(t.nrows, tf) == len(tf)
    assert compute(t.a.nrows, tf) == len(tf)

    with pytest.raises(ValueError):
        compute(t.nelements(axis=1), tf)

    with pytest.raises(ValueError):
        compute(t.a.nelements(axis=1), tf)


def test_nunique_sframe(t, tf):
    assert compute(t.nunique(), tf) == tf.dropna().unique().num_rows()


def test_nunique_sarray(t, tf):
    assert compute(t.a.nunique(), tf) == tf['a'].dropna().unique().size()


def test_distinct_sframe(t, tf):
    expr = t.distinct()
    result = compute(expr, tf)
    expected = tf.unique()
    tm.assert_frame_equal(odo(result, pd.DataFrame),
                          odo(expected, pd.DataFrame))


def test_distinct_sarray(t, tf):
    expr = t.b.distinct()
    result = compute(expr, tf)
    expected = tf['b'].unique()
    tm.assert_series_equal(odo(result, pd.Series), odo(expected, pd.Series))


@pytest.mark.parametrize('freq',
                         ['year', 'month', 'day', 'hour', 'minute', 'second'])
def test_datetime_attr(d, df, freq):
    expr = getattr(d.dates, freq)
    result = compute(expr, df)
    expected = df['dates'].split_datetime('', limit=[freq])[freq]
    tm.assert_series_equal(odo(result, pd.Series), odo(expected, pd.Series))


@pytest.mark.xfail(raises=(TypeError, AssertionError),
                   reason='No micro- or milliseconds yet')
@pytest.mark.parametrize('freq', ['millisecond', 'microsecond'])
def test_datetime_attr_high_precision(d, dfu, freq):
    result = compute(getattr(d.dates, freq), dfu)
    expected = dfu['dates'].split_datetime('', limit=[freq])[freq]
    assert any(expected)
    tm.assert_series_equal(odo(result, pd.Series), odo(expected, pd.Series))


def test_transform(t, tf):
    expr = transform(t, foo=t.a + t.a)
    result = compute(expr, tf)
    expected = gl.SFrame(tf).add_column(tf['a'] + tf['a'], name='foo')
    tm.assert_frame_equal(odo(result, pd.DataFrame),
                          odo(expected, pd.DataFrame))


def test_relabel_sframe(t, tf):
    result = compute(t.relabel(a='d', b='e'), tf)
    expected = tf.rename(dict(a='d', b='e'))
    tm.assert_frame_equal(odo(result, pd.DataFrame),
                          odo(expected, pd.DataFrame))


def test_label_does_nothing_on_sarray(t, tf):
    assert all(compute(t.a.label('b'), tf) == tf['a'])
