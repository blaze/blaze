from __future__ import absolute_import, division, print_function

import sys

from blaze.expr.scalar import *
from blaze.compatibility import xfail, basestring
from blaze.utils import raises
from datetime import date, datetime

import pytest

x = ScalarSymbol('x')
y = ScalarSymbol('y')

def test_basic():
    expr = (x + y) * 3

    assert eval(str(expr)) == expr
    assert expr == Mul(Add(ScalarSymbol('x'), ScalarSymbol('y')), 3)


def test_eval_str():
    expr = (x + y) * 3
    assert expr.eval_str() == '(x + y) * 3'

    assert eval_str(1) == '1'
    assert eval_str('Alice') == "'Alice'"

    print(eval_str(-x))
    assert eval_str(-x) == '-x'


def test_str():
    x = ScalarSymbol('x', 'real')

    assert str(x + 10) == 'x + 10'


def ishashable(x):
    try:
        hash(x)
        return True
    except:
        return False


def test_ScalarSymbol_is_hashable():
    assert ishashable(x)


def test_relationals():
    x = ScalarSymbol('x', 'real')
    for expr in [x < 1, x > 1, x == 1, x != 1, x <= 1, x >= 1]:
        assert expr.dshape == dshape('bool')
        assert eval(str(expr)) == expr


def test_numbers():
    x = ScalarSymbol('x', 'real')
    y = ScalarSymbol('x', 'int')
    for expr in [x + 1, x - 1, x * 1, x + y, x - y, x / y, x * y + x + y,
                 x**y, x**2, 2**x, x % 5, -x,
                 sin(x), cos(x ** 2), exp(log(y))]:
        assert expr.dshape == dshape('real')
        assert eval(str(expr)) == expr

    assert (-y).dshape == dshape('int')


@xfail(reason="TODO")
def test_neg_dshape_unsigned():
    y = ScalarSymbol('x', 'uint32')
    assert (-y).dshape == dshape('int32')


@xfail(reason="TODO")
def test_arithmetic_dshape_inference():
    x = ScalarSymbol('x', 'int')
    y = ScalarSymbol('y', 'int')
    assert (x + y).dshape == dshape('int')


def test_date_coercion():
    d = ScalarSymbol('d', 'date')
    expr = d < '2012-01-01'
    assert isinstance(expr.rhs, date)


def test_datetime_coercion():
    d = ScalarSymbol('d', 'datetime')
    expr = d > '2012-01-01T12:30:00'
    assert isinstance(expr.rhs, datetime)


class TestExprify(object):
    dtypes = {'x': 'int', 'y': 'real', 'z': 'int32'}
    x = ScalarSymbol('x', 'int')
    y = ScalarSymbol('y', 'real')
    z = ScalarSymbol('z', 'int32')
    name = ScalarSymbol('name', 'string')

    def test_basic_arithmetic(self):
        assert exprify('x + y', self.dtypes).isidentical(self.x + self.y)

        other = isnan(sin(x) + y)
        assert exprify('isnan(sin(x) + y)', self.dtypes).isidentical(other)

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
        dtypes = {'x': 'int', 'y': 'real', 'z': 'int32'}

        with pytest.raises(AssertionError):
            exprify('x < y < z', dtypes)

        with pytest.raises(NotImplementedError):
            exprify('os.listdir()', {})

        with pytest.raises(NotImplementedError):
            exprify('os.listdir()', {'os': 'int', 'os.listdir': 'real'})

        with pytest.raises(ValueError):
            exprify('__x + __y', dtypes)

        with pytest.raises(NotImplementedError):
            exprify('y if x else y', dtypes)

        with pytest.raises(NotImplementedError):
            exprify('lambda x, y: x + y', dtypes)

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
    assert raises(TypeError, lambda: scalar_coerce('{x: int, y: int}', '1'))
    assert raises(TypeError, lambda: scalar_coerce('3 * int', '1'))

    assert scalar_coerce('date', 'Jan 1st, 2012') == date(2012, 1, 1)
    assert scalar_coerce('datetime', 'Jan 1st, 2012 12:00:00') == \
            datetime(2012, 1, 1, 12, 0, 0)
    assert raises(ValueError,
                  lambda: scalar_coerce('date', 'Jan 1st, 2012 12:00:00'))
