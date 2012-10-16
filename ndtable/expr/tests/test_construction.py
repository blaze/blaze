from ndtable.expr.nodes import Node
from ndtable.expr.graph import NDTable, ScalarNode
from ndtable.expr.viz import dump, build_graph

def test_scalar_arguments():
    a = NDTable([1,2,3])
    children = a.node.children

    assert len(children) == 3

def test_dynamic_arguments():
    a = NDTable([])
    b = NDTable([a])

    children = b.node.children
    assert len(children) == 1

def test_dynamic_explicit():
    a = NDTable([])
    b = NDTable([a], depends=[a])

    children = b.node.children
    assert len(children) == 1

def test_binary_ops():
    a = NDTable([])
    b = NDTable([])

    x = a+b
    y = x*a

    dump(y, filename='binary')

def test_unary_ops():
    a = NDTable([])

    x = abs(a)
    dump(x, filename='unary')

def test_indexing():
    a = NDTable([])

    x = a[0]
    dump(x, filename='indexer')

def test_slice():
    a = NDTable([])

    x = a[0:1]
    dump(x, filename='indexer')

def test_scalars():
    a = ScalarNode(1)
    b = ScalarNode(1)

    x = NDTable([a,b])
    dump(x, filename='scalars')
