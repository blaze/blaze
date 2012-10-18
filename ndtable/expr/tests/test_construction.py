from ndtable.datashape.coretypes import float64
from ndtable.expr.graph import IntNode, FloatNode
from ndtable.expr.nodes import Node, traverse
from ndtable.expr.viz import dump
from ndtable.table import NDTable
from ndtable.expr.graph import App


# Print out graphviz to the screen
DEBUG = False

def test_walk():
    e = Node([])
    d = Node([])
    b = Node([d])
    c = Node([e])
    a = Node([b,c])

    assert len([n for n in a]) == 4

    if DEBUG:
        dump(a, filename='walk', tree=True)

def test_traverse():
    e = Node([])
    d = Node([])
    b = Node([d])
    c = Node([e])
    a = Node([b,c])

    [n for n in traverse(a)]

    if DEBUG:
        dump(a, filename='walk', tree=True)

def test_scalar_arguments():
    a = NDTable([1,2,3])
    children = a.children

    assert len(children) == 3

def test_dynamic_arguments():
    a = NDTable([])
    b = NDTable([a])

    children = b.children
    assert len(children) == 1

def test_binary_ops():
    a = IntNode(1)
    b = IntNode(2)

    x = a+b

    if DEBUG:
        dump(x, filename='binary')

def test_binary_mixed():
    a = IntNode(1)

    x = a+2

    if DEBUG:
        dump(x, filename='binarymixed')

def test_unary_ops():
    a = IntNode(1)

    x = abs(a)

    if DEBUG:
        dump(x, filename='unary')

def test_indexing():
    a = NDTable([])

    x = a[0]

    if DEBUG:
        dump(x, filename='indexer')

def test_slice():
    a = NDTable([])

    x = a[0:1]

    if DEBUG:
        dump(x, filename='slice')

def test_scalars():
    a = IntNode(1)
    b = IntNode(1)
    c = IntNode(2)

    x = abs((a + b + c + 3) * 4)

    if DEBUG:
        dump(x, filename='scalars')

def test_preserve_types():
    a = IntNode(1)
    b = FloatNode(1.0)

    x = a + b
    assert isinstance(x, App)
    assert x.dom == [float64, float64]
    assert x.cod == float64
