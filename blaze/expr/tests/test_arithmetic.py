from blaze.expr import *

x = Symbol('x', '5 * 3 * int32')
y = Symbol('y', '5 * 3 * int32')
a = Symbol('a', 'int32')

b = Symbol('b', '5 * 3 * bool')

def test_arithmetic_dshape_on_collections():
    assert Add(x, y).shape == x.shape == y.shape

def test_unary_ops_are_elemwise():
    assert USub(x).shape == x.shape
    assert Not(b).shape == b.shape
