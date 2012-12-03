from ndtable.expr import ops
from ndtable.table import NDArray
from ndtable.expr.graph import IntNode, FloatNode, VAL, OP, APP
from ndtable.engine.pipeline import toposort, topops, topovals, Pipeline
from pprint import pprint

DEBUG = False

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

    # Add(1,Mul(2,Abs(3.0)))
    if DEBUG:
        pprint(plan, width=1)

    plan = line.run_pipeline(y)

    # Add(Add(1,Add(2,3.0)),Add(1,Mul(2,Abs(3.0))))
    if DEBUG:
        pprint(plan, width=1)

    plan = line.run_pipeline(x+y)

    # Add(Mul(Add(1,Add(2,3.0)),Add(Add(1,Mul(2,Abs(3.0))),2)),3)
    if DEBUG:
        pprint(plan, width=1)

    plan = line.run_pipeline(x*(y+2)+3)

    # Add(Array(39558864){dshape("3 int64")},Array(39558864){dshape("3 int64")})
    if DEBUG:
        pprint(plan, width=1)

    plan = line.run_pipeline(d+d)

    # Add(
    #   Add(Array(39558864){dshape("3 int64")}, Array(39558864){dshape("3 int64")})
    # , Add(Array(39558864){dshape("3 int64")}, Array(39558864){dshape("3 int64")})
    # )
    if DEBUG:
        pprint(plan, width=1)

    plan = line.run_pipeline((d+d)+(d+d))

    if DEBUG:
        pprint(plan, width=1)


    # Mul(
    #   Mul(Array(39558864){dshape("3 int64")}, Array(39558864){dshape("3 int64")})
    # , Mul(Array(39558864){dshape("3 int64")}, Array(39558864){dshape("3 int64")})
    # )
    plan = line.run_pipeline((d*d)*(d*d))

    if DEBUG:
        pprint(plan, width=1)
