from ndtable.expr.deferred import DeferredTable

def test_scalar_arguments():
    a = DeferredTable([1,2,3])
    fields = a.node.fields

    assert fields[0] == 'init'
    assert len(fields[1:4]) == 3
