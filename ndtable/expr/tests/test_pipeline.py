from ndtable.expr import ops
from ndtable.table import NDArray
from ndtable.expr.graph import IntNode, FloatNode, VAL, OP, APP
from ndtable.engine.pipeline import toposort, topops, topovals, Pipeline
from pprint import pprint

DEBUG = True

#------------------------------------------------------------------------
# Sample Graph
#------------------------------------------------------------------------

a = IntNode(1)
b = IntNode(2)
c = FloatNode(3.0)

x = a+(b+c)
y = a+(b*abs(c))

d = NDArray([1,2,3])

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

    assert lst[0].__class__ == ops.Abs
    assert lst[1].__class__ == ops.Mul
    assert lst[2].__class__ == ops.Add

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

    if DEBUG:
        print pprint(plan, width=1)

    plan = line.run_pipeline(y)

    if DEBUG:
        print pprint(plan, width=1)

    plan = line.run_pipeline(x+y)

    if DEBUG:
        print pprint(plan, width=1)

    plan = line.run_pipeline(x*(y+2)+3)

    if DEBUG:
        print pprint(plan, width=1)

    plan = line.run_pipeline(d+d)

    if DEBUG:
        print pprint(plan, width=1)

    plan = line.run_pipeline((d+d)+(d+d))

    if DEBUG:
        print pprint(plan, width=1)

    plan = line.run_pipeline((d*d)*(d*d))

    if DEBUG:
        print pprint(plan, width=1)
