from blaze.expr import *
from datashape import dshape

x = Symbol('x', '5 * 3 * int32')
y = Symbol('y', '5 * 3 * int32')
a = Symbol('a', 'int32')

b = Symbol('b', '5 * 3 * bool')

def test_arithmetic_dshape_on_collections():
    assert Add(x, y).shape == x.shape == y.shape

def test_unary_ops_are_elemwise():
    assert USub(x).shape == x.shape
    assert Not(b).shape == b.shape

def test_relations_maintain_shape():
    assert Gt(x, y).shape == x.shape

def test_relations_are_boolean():
    assert Gt(x, y).schema == dshape('bool')
