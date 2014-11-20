from blaze.expr import *
from blaze.expr.math import *

x = symbol('x', '5 * 3 * int')

def test_math_shapes():
    assert sin(x).shape == x.shape
