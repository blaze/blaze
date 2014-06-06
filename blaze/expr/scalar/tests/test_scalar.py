from blaze.expr.scalar import *
import math

x = NumberSymbol('x')
y = NumberSymbol('y')

def test_basic():
    expr = (x + y) * 3

    assert eval(str(expr)) == expr
    assert expr == Mul(Add(NumberSymbol('x'), NumberSymbol('y')), 3)
