from __future__ import absolute_import, division, print_function

import pytest
import sys

from blaze.expr.arithmetic import (scalar_coerce, Mult, Add, dshape)
from blaze.expr.math import sin, cos, isnan, exp, log
from blaze.expr import symbol, eval_str
from blaze.compatibility import xfail, basestring
from datetime import date, datetime

x = symbol('x', 'real')
y = symbol('y', 'real')
b = symbol('b', 'bool')


def test_basic():
    expr = (x + y) * 3

    assert eval(str(expr)).isidentical(expr)
    assert expr.isidentical(
        Mult(Add(symbol('x', 'real'), symbol('y', 'real')), 3),
    )


def test_eval_str():
    expr = (x + y) * 3
    assert eval_str(expr) == '(x + y) * 3'

    assert eval_str(1) == '1'
    assert eval_str('Alice') == "'Alice'"
    assert "'Alice'" in eval_str(u'Alice')

    print(eval_str(-x))
    assert eval_str(-x) == '-x'

    assert '~' in eval_str(~b)


def test_str():
    x = symbol('x', 'real')
    assert str(x + 10) == 'x + 10'


def test_invert():
    x = symbol('x', 'bool')
    expr = ~x
    assert expr.op(x).isidentical(expr)


def test_boolean_math_has_boolean_methods():
    x = symbol('x', '?int')
    expr = ~(isnan(x)) | (x > 0)

    assert eval(str(expr)).isidentical(expr)


def ishashable(x):
    try:
        hash(x)
        return True
    except:
        return False


def test_Symbol_is_hashable():
    assert ishashable(x)


def test_relationals():
    x = symbol('x', 'real')
    for expr in [x < 1, x > 1, x == 1, x != 1, x <= 1, x >= 1, ~b]:
        assert expr.dshape == dshape('bool')
        assert eval(str(expr)).isidentical(expr)


def test_numbers():
    x = symbol('x', 'real')
    y = symbol('y', 'int')
    for expr in [x + 1, x - 1, x * 1, x + y, x - y, x / y, x * y + x + y,
                 x**y, x**2, 2**x, x % 5, -x,
                 sin(x), cos(x ** 2), exp(log(y))]:
        assert expr.dshape == dshape('real')
        assert eval(str(expr)).isidentical(expr)

    assert (-y).dshape == dshape('int')


@xfail(reason="TODO")
def test_neg_dshape_unsigned():
    y = symbol('y', 'uint32')
    assert (-y).dshape == dshape('int32')


def test_arithmetic_dshape_inference():
    x = symbol('x', 'int')
    y = symbol('y', 'int')
    assert (x + y).dshape == dshape('int')


def test_date_coercion():
    d = symbol('d', 'date')
    expr = d < '2012-01-01'
    assert isinstance(expr.rhs, date)


def test_datetime_coercion():
    d = symbol('d', 'datetime')
    expr = d > '2012-01-01T12:30:00'
    assert isinstance(expr.rhs, datetime)


def test_scalar_coerce():
    assert scalar_coerce('int', 1) == 1
    assert scalar_coerce('int', '1') == 1
    assert scalar_coerce('{x: int}', '1') == 1
    with pytest.raises(TypeError):
        scalar_coerce('{x: int, y: int}', '1')

    assert scalar_coerce('date', 'Jan 1st, 2012') == date(2012, 1, 1)

    assert (scalar_coerce('datetime', 'Jan 1st, 2012 12:00:00') ==
            datetime(2012, 1, 1, 12, 0, 0))

    with pytest.raises(TypeError):
        scalar_coerce('date', 'Jan 1st, 2012 12:00:00')

    assert scalar_coerce('?date', 'Jan 1st, 2012') == date(2012, 1, 1)
    assert scalar_coerce('?date', '2012-12-01') == date(2012, 12, 1)
    with pytest.raises(TypeError):
        scalar_coerce('?date', '')
    assert scalar_coerce('?int', 0) == 0
    assert scalar_coerce('?int', '0') == 0
    x = symbol('x', '?int')
    assert scalar_coerce('?int', x) is x


def test_scalar_name_dtype():
    x = symbol('x', 'int64')
    assert x._name == 'x'
    assert x.dshape == dshape('int64')


def test_scalar_field():
    x = symbol('x', '{name: string, amount: int64, when: datetime}')
    assert 'amount' in dir(x)
    assert x.amount.dshape == dshape('int64')


def test_scalar_projection():
    x = symbol('x', '{name: string, amount: int64, when: datetime}')
    assert x[['amount', 'when']].dshape == \
            dshape('{amount: int64, when: datetime}')
