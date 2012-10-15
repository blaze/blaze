from ndtable.expr.deferred import DeferredTable
from ndtable.expr.viz import view, build_graph

def test_scalar_arguments():
    a = DeferredTable([1,2,3])
    fields = a.node.fields

    assert fields[0] == 'init'
    assert len(fields[1:]) == 3

def test_dynamic_arguments():
    a = DeferredTable([])
    b = DeferredTable([a])

    fields = b.node.fields
    assert fields[0] == 'init'
    assert len(fields[1:]) == 1

def test_dynamic_explicit():
    a = DeferredTable([])
    b = DeferredTable([a], depends=[a])

    fields = b.node.fields
    assert fields[0] == 'init'
    assert len(b.node.listeners) > 0

def test_simple_ops():
    a = DeferredTable([])
    b = DeferredTable([])

    c = a+b
    d = c*b

    print d.fields
    print d.listeners

    _, graph = build_graph(d)
    view('simple', graph)
