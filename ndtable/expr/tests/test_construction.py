from ndtable.expr.deferred import DeferredTable
from ndtable.expr.nodes import Node, ScalarNode
from ndtable.expr.viz import dump, build_graph

def test_scalar_arguments():
    a = DeferredTable([1,2,3])
    children = a.node.children

    assert len(children) == 3

def test_dynamic_arguments():
    a = DeferredTable([])
    b = DeferredTable([a])

    children = b.node.children
    assert len(children) == 1

def test_dynamic_explicit():
    a = DeferredTable([])
    b = DeferredTable([a], depends=[a])

    children = b.node.children
    assert len(children) == 1

def test_binary_ops():
    a = DeferredTable([])
    b = DeferredTable([])

    x = a+b
    y = x*b

    dump(y, filename='binary')

def test_unary_ops():
    a = DeferredTable([])

    x = abs(a)
    dump(x, filename='unary')

def test_indexing():
    a = DeferredTable([])

    x = a[0]
    dump(x, filename='indexer')

def test_slice():
    a = DeferredTable([])

    x = a[0:1]
    dump(x, filename='indexer')

def test_scalars():
    a = ScalarNode(1)
    b = ScalarNode(1)

    x = DeferredTable([a,b])
    dump(x, filename='indexer')
