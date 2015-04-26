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
from blaze import by, compute, symbol, discover, join


@pytest.fixture
def t(tf):
    return symbol('t', discover(tf))


@pytest.fixture
def tf():
    return gl.SFrame(
        pd.DataFrame({'a': [1, 2, 3, 4, 5, 6], 'b': list('abcabc'),
                      'c': np.random.randn(6)}))


@pytest.fixture
def s(sf):
    return symbol('s', discover(sf))


@pytest.fixture
def sf():
    return gl.SFrame(
        pd.DataFrame({'a': [1, 2, 3, 3, 1, 1, 2, 3, 4, 4, 3, 1],
                      'b': list('bacbaccaacbb'),
                      'c': np.random.randn(12)}))


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
