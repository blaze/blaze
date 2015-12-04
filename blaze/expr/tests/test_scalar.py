from __future__ import absolute_import, division, print_function

import pytest
import sys

from blaze.expr.arithmetic import (scalar_coerce, Mult, Add, dshape)
from blaze.expr.math import sin, cos, isnan, exp, log
from blaze.expr import symbol, eval_str, exprify
from blaze.compatibility import xfail, basestring
from datetime import date, datetime

x = symbol('x', 'real')
y = symbol('y', 'real')
b = symbol('b', 'bool')


def test_basic():
    expr = (x + y) * 3

    assert eval(str(expr)).isidentical(expr)
    assert expr.isidentical(Mult(Add(symbol('x', 'real'), symbol('y', 'real')), 3))


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


class TestExprify(object):
    dtypes = {'x': 'int', 'y': 'real', 'z': 'int32'}
    x = symbol('x', 'int')
    y = symbol('y', 'real')
    z = symbol('z', 'int32')
    name = symbol('name', 'string')

    def test_basic_arithmetic(self):
        assert exprify('x + y', self.dtypes).isidentical(self.x + self.y)

        expected = isnan(sin(x) + y)
        result = exprify('isnan(sin(x) + y)', self.dtypes)
        assert str(expected) == str(result)

        # parsed as a Num in Python 2 and a UnaryOp in Python 3
        assert exprify('-1', {}) == -1

        # parsed as UnaryOp(op=USub(), operand=1)
        assert exprify('-x', self.dtypes).isidentical(-self.x)

        assert exprify('-x + y', self.dtypes).isidentical(-self.x + self.y)

        other = self.x * self.y + self.z
        assert exprify('x * y + z', self.dtypes).isidentical(other)
        assert exprify('x ** y', self.dtypes).isidentical(self.x ** self.y)

        other = self.x / self.y / self.z + 1
        assert exprify('x / y / z + 1', self.dtypes).isidentical(other)

        other = self.x / self.y % self.z + 2 ** self.y
        assert exprify('x / y % z + 2 ** y', self.dtypes).isidentical(other)

        assert exprify('x // y', self.dtypes).isidentical(self.x // self.y)
        assert exprify('1 // y // x', self.dtypes).isidentical(
            1 // self.y // self.x)

    def test_comparison(self):
        other = (self.x == 1) | (self.x == 2)
        assert exprify('(x == 1) | (x == 2)', self.dtypes).isidentical(other)

    def test_simple_boolean_not(self):
        x = symbol('x', 'bool')
        assert exprify('~x', {'x': 'bool'}).isidentical(~x)

    def test_literal_string_compare(self):
        other = self.name == "Alice"
        result = exprify('name == "Alice"', {'name': 'string'})
        assert isinstance(result.rhs, basestring)
        assert result.isidentical(other)

    def test_literal_int_compare(self):
        other = self.x == 1
        result = exprify('x == 1', self.dtypes)
        assert isinstance(result.rhs, int)
        assert result.isidentical(other)

    def test_literal_float_compare(self):
        other = self.y == 1.0
        result = exprify('y == 1.0', self.dtypes)
        assert isinstance(result.rhs, float)
        assert result.isidentical(other)

    def test_failing_exprify(self):
        dtypes = self.dtypes

        with pytest.raises(AssertionError):
            exprify('x < y < z', dtypes)

        with pytest.raises(NotImplementedError):
            exprify('os.listdir()', {})

        with pytest.raises(NotImplementedError):
            exprify('os.listdir()', {'os': 'int', 'os.listdir': 'real'})

        with pytest.raises(ValueError):
            exprify('__x + __y', {'__x': 'int', '__y': 'real'})

        with pytest.raises(NotImplementedError):
            exprify('y if x else y', dtypes)

    def test_functiondef_fail(self):
        dtypes = self.dtypes
        with pytest.raises(NotImplementedError):
            exprify('lambda x, y: x + y', dtypes)

        with pytest.raises(SyntaxError):
            exprify('def f(x): return x', dtypes={'x': 'int'})

    def test_comprehensions_fail(self):
        dtypes = self.dtypes

        if sys.version_info < (2, 7):
            # dict and set comprehensions are not implemented in Python < 2.7
            error = SyntaxError
        else:
            # and we don't allow them in versions that do
            error = NotImplementedError

        with pytest.raises(error):
            exprify('{x: y for z in y}', dtypes)

        with pytest.raises(error):
            exprify('{x for z in y}', dtypes)

        with pytest.raises(NotImplementedError):
            exprify('[x for z in y]', dtypes)

        with pytest.raises(NotImplementedError):
            exprify('(x for y in z)', dtypes)

    def test_boolop_fails(self):
        dtypes = self.dtypes

        with pytest.raises(NotImplementedError):
            exprify('x or y', dtypes)

        with pytest.raises(NotImplementedError):
            exprify('x and y', dtypes)

        with pytest.raises(NotImplementedError):
            exprify('not x', dtypes)

    def test_scope(self):
        dtypes = {'sin': 'int'}
        with pytest.raises(ValueError):
            exprify('sin + 1', dtypes)

        with pytest.raises(TypeError):
            sin + 1


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
