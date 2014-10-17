from blaze.expr import *
from datashape import dshape

x = Symbol('x', '5 * 3 * int32')
y = Symbol('y', '5 * 3 * int32')
a = Symbol('a', 'int32')

b = Symbol('b', '5 * 3 * bool')

def test_arithmetic_dshape_on_collections():
    assert Add(x, y).shape == x.shape == y.shape

def test_arithmetic_broadcasts_to_scalars():
    assert Add(x, a).shape == x.shape
    assert Add(x, 1).shape == x.shape

def test_unary_ops_are_elemwise():
    assert USub(x).shape == x.shape
    assert Not(b).shape == b.shape

def test_relations_maintain_shape():
    assert Gt(x, y).shape == x.shape

def test_relations_are_boolean():
    assert Gt(x, y).schema == dshape('bool')

def test_names():
    assert Add(x, 1)._name == x._name
    assert Add(1, x)._name == x._name
    assert Mult(Add(1, x), 2)._name == x._name

    assert Add(y, x)._name != x._name
    assert Add(y, x)._name != y._name
