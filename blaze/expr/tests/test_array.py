from blaze.expr import *


x = symbol('x', '4 * 5 * int32')
y = symbol('y', '5 * int32')
z = symbol('z', '4 * int32')

def test_shape_of_binop_of_differently_shaped_arrays():
    assert (x + y).shape == x.shape
