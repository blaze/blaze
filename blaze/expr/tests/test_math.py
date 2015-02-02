from blaze.expr import *
from blaze.expr.math import *
from blaze.expr.math import abs as mathabs

x = symbol('x', '5 * 3 * int')

def test_math_shapes():
    assert sin(x).shape == x.shape

def test_abs():
    assert abs(x).isidentical(mathabs(x))
