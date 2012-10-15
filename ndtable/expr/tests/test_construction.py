from ndtable.expr.deferred import DeferredTable

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
    b = DeferredTable([a], depends=a)

    fields = b.node.fields
    assert fields[0] == 'init'
    assert len(b.node.listeners) > 0
