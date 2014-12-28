from blaze.expr import *
from blaze.expr.arrays import *


w = symbol('x', '5 * 6 * int32')
x = symbol('x', '4 * 5 * int32')
y = symbol('y', '5 * int32')
z = symbol('z', '4 * int32')

def test_shape_of_binop_of_differently_shaped_arrays():
    assert (x + y).shape == x.shape


def test_transpose():
    assert ndim(x.dshape) == 2

    assert x.T.shape == (5, 4)


def test_dot():
    assert x.dot(y).shape == (4,)
    assert x.dot(w).shape == (4, 6)


def test_tensordot():
    expr = tensordot(x, z, axes=[0, 0])
    assert expr.shape == (5,)

    expr = tensordot(x, z, axes=[[0], [0]])
    assert expr.shape == (5,)
