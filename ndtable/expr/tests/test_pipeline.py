from ndtable.expr import ops
from ndtable.expr.graph import IntNode, FloatNode, VAL, OP, APP
from ndtable.engine.pipeline import toposort, topops, topovals, Pipeline

#------------------------------------------------------------------------
# Sample Graph
#------------------------------------------------------------------------

a = IntNode(1)
b = IntNode(2)
c = FloatNode(3.0)

x = a+(b+c)
y = a+(b*abs(c))

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

def test_simple_sort():
    lst = toposort(lambda x: True, x)
    assert len(lst) == 6


def test_simple_sort_ops():
    lst = topops(y)
    # We expect this:

    #  add
    #  / \
    # 1  mul
    #    / \
    #   2   abs
    #        |
    #       3.0

    # To collapse into this:

    #    abs
    #     |
    #    mul
    #     |
    #    add

    assert lst[0].__class__ == ops.abs
    assert lst[1].__class__ == ops.mul
    assert lst[2].__class__ == ops.add

    assert lst[0].kind == OP
    assert lst[1].kind == OP
    assert lst[2].kind == OP


def test_simple_sort_vals():
    lst = topovals(y)
    # We expect this:

    #  add
    #  / \
    # 1  mul
    #    / \
    #   2   abs
    #        |
    #       3.0

    # To collapse into this:

    #    1
    #    |
    #    2
    #    |
    #   3.0

    assert lst[0].val == 1
    assert lst[1].val == 2
    assert lst[2].val == 3.0

def test_simple_pipeline():
    line = Pipeline()
    plan = line.run_pipeline(x)
    print plan
