from blaze.expr.scalar import *
from blaze.compatibility import skip
import math

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
        print expr
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


@skip("TODO")
def test_neg_dshape_unsigned():
    y = ScalarSymbol('x', 'uint32')
    assert (-y).dshape == dshape('int32')
