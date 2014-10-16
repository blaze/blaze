from blaze.expr import *

x = Symbol('x', '5 * 3 * int32')
y = Symbol('y', '5 * 3 * int32')
a = Symbol('a', 'int32')

def test_arithmetic_dshape_on_collections():
    assert Add(x, y).shape == (5, 3)
