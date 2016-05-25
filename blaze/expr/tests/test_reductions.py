from itertools import product

import pytest

from blaze.expr import symbol, summary
from datashape import dshape
from datashape.util.testing import assert_dshape_equal


def test_reduction_dshape():
    x = symbol('x', '5 * 3 * float32')
    assert x.sum().dshape == dshape('float64')
    assert x.sum(axis=0).dshape == dshape('3 * float64')
    assert x.sum(axis=1).dshape == dshape('5 * float64')
    assert x.sum(axis=(0, 1)).dshape == dshape('float64')


def test_keepdims():
    x = symbol('x', '5 * 3 * float32')
    assert x.sum(axis=0, keepdims=True).dshape == dshape('1 * 3 * float64')
    assert x.sum(axis=1, keepdims=True).dshape == dshape('5 * 1 * float64')
    assert x.sum(axis=(0, 1), keepdims=True).dshape == dshape(
        '1 * 1 * float64')

    assert x.std(axis=0, keepdims=True).shape == (1, 3)


def test_summary_keepdims():
    x = symbol('x', '5 * 3 * float32')
    assert summary(a=x.min(), b=x.max()).dshape == \
        dshape('{a: float32, b: float32}')
    assert summary(a=x.min(), b=x.max(), keepdims=True).dshape == \
        dshape('1 * 1 * {a: float32, b: float32}')


def test_summary_axis():
    x = symbol('x', '5 * 3 * float32')
    assert summary(a=x.min(), b=x.max(), axis=0).dshape == \
        dshape('3 * {a: float32, b: float32}')
    assert summary(a=x.min(), b=x.max(), axis=1).dshape == \
        dshape('5 * {a: float32, b: float32}')
    assert summary(a=x.min(), b=x.max(), axis=1, keepdims=True).dshape == \
        dshape('5 * 1 * {a: float32, b: float32}')


def test_summary_str():
    x = symbol('x', '5 * 3 * float32')
    assert 'keepdims' not in str(summary(a=x.min(), b=x.max()))


def test_axis_kwarg_is_normalized_to_tuple():
    x = symbol('x', '5 * 3 * float32')
    exprs = [x.sum(), x.sum(axis=1), x.sum(axis=[1]), x.std(), x.mean(axis=1)]
    for expr in exprs:
        assert isinstance(expr.axis, tuple)


def test_summary_with_multiple_children():
    t = symbol('t', 'var * {x: int, y: int, z: int}')

    assert summary(a=t.x.sum() + t.y.sum())._child.isidentical(t)


def test_dir():
    t = symbol('t', '10 * int')
    assert 'mean' in dir(t)

    t = symbol('t', 'int')
    assert 'mean' not in dir(t)


def test_norms():
    x = symbol('x', '5 * 3 * float32')
    assert x.vnorm().isidentical(x.vnorm('fro'))
    assert x.vnorm().isidentical(x.vnorm(2))
    assert x.vnorm(axis=0).shape == (3,)
    assert x.vnorm(axis=0, keepdims=True).shape == (1, 3)


@pytest.mark.parametrize('reduc', ['max', 'min', 'sum', 'mean', 'std', 'var'])
def test_reductions_on_record_dshape(reduc):
    t = symbol('t', '10 * {a: int64, b: string}')
    with pytest.raises(AttributeError):
        getattr(t, reduc)


@pytest.mark.parametrize('reduc', ['max', 'min', 'sum', 'mean', 'std', 'var'])
def test_boolean_has_reductions(reduc):
    assert hasattr(symbol('t', 'var * bool'), reduc)


@pytest.mark.parametrize(['reduc', 'measure'],
                         product(['max', 'min'],
                                 ['date', 'datetime', 'timedelta']))
def test_max_min_on_datetime_and_timedelta(reduc, measure):
    assert hasattr(symbol('t', 'var * %s' % measure), reduc)


def test_reduction_naming_with_generated_leaves():
    assert symbol('_', 'var * float64').sum()._name == 'sum'


@pytest.mark.parametrize('func', ['sum', 'mean', 'std', 'var'])
def test_decimal_reduction(func):
    t = symbol('t', 'var * decimal[11, 2]')
    method = getattr(t, func)
    assert_dshape_equal(method().dshape, dshape("decimal[11, 2]"))


@pytest.mark.parametrize('func', ['sum', 'mean', 'std', 'var'])
def test_timedelta_reduction(func):
    t = symbol('t', 'var * timedelta')
    method = getattr(t, func)
    assert_dshape_equal(method().dshape, dshape("timedelta"))
