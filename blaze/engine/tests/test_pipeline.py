# TODO assertsion for the output'd values... right now they have
# pointer addresses so its hard to do string matching

from blaze.expr import ops
from blaze.table import NDArray
from blaze.expr.graph import IntNode, FloatNode, VAL, OP, APP
from blaze.engine.pipeline import toposort, topops, topovals, Pipeline
from pprint import pprint
from difflib import ndiff

from unittest import skip

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

@skip
def test_simple_sort():
    lst = toposort(lambda x: True, x)
    assert len(lst) == 6

@skip
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
    _, plan = line.run_pipeline(x)

    # Add(1,Mul(2,Abs(3.0)))
    if DEBUG:
        pprint(plan, width=1)

    ctx, plan = line.run_pipeline(y)

    # Add(Add(1,Add(2,3.0)),Add(1,Mul(2,Abs(3.0))))
    if DEBUG:
        pprint(plan, width=1)

    ctx, plan = line.run_pipeline(x+y)

    # ATerm
    # -----
    # Add(Mul(Add(1,Add(2,3.0)),Add(Add(1,Mul(2,Abs(3.0))),2)),3)

    # Instructions
    # ------------
    #   %0 = <ufunc 'add'> const(2) const(3.0)
    #   %1 = <ufunc 'add'> const(1) '%0'
    #   %2 = <ufunc 'absolute'> const(3.0)
    #   %3 = <ufunc 'multiply'> const(2) '%2'
    #   %4 = <ufunc 'add'> const(1) '%3'
    #   %5 = <ufunc 'add'> '%4' const(2)
    #   %6 = <ufunc 'multiply'> '%1' '%5'
    #   %7 = <ufunc 'add'> '%6' const(3)

    if DEBUG:
        pprint(plan, width=1)

    ctx, plan = line.run_pipeline(x*(y+2)+3)

    #------------------------------------------------------------------------

    # Add(Array(39558864){dshape("3 int64")},Array(39558864){dshape("3 int64")})
    if DEBUG:
        pprint(plan, width=1)

    ctx, plan = line.run_pipeline(d+d)

    # Add(
    #   Add(Array(39558864){dshape("3 int64")}, Array(39558864){dshape("3 int64")})
    # , Add(Array(39558864){dshape("3 int64")}, Array(39558864){dshape("3 int64")})
    # )
    if DEBUG:
        pprint(plan, width=1)

    ctx, plan = line.run_pipeline((d+d)+(d+d))

    if DEBUG:
        pprint(plan, width=1)


    # Mul(
    #   Mul(Array(39558864){dshape("3 int64")}, Array(39558864){dshape("3 int64")})
    # , Mul(Array(39558864){dshape("3 int64")}, Array(39558864){dshape("3 int64")})
    # )
    ctx, plan = line.run_pipeline((d*d)*(d*d))

    if DEBUG:
        pprint(plan, width=1)

def test_complex_pipeline1():
    a = NDArray([1,2,3])
    b = NDArray([1,2,3])
    c = NDArray([1,2,3])
    d = NDArray([1,2,3])

    line = Pipeline()
    _, plan = line.run_pipeline(((a*b)+(c*d))**2)

    # Pow(
    #   Arithmetic(
    #     Add
    #   , Arithmetic(
    #       Mul
    #     , Array(){dshape("3,int64"), 40127160}
    #     , Array(){dshape("3,int64"), 40015272}
    #     ){dshape("int64"), 40076400}
    #   , Arithmetic(
    #       Mul
    #     , Array{dshape("3,int64"), 40069816}
    #     , Array{dshape("3,int64"), 40080448}
    #     ){dshape("int64"), 40076016}
    #   ){dshape("int64"), 40077552}
    # , 2{dshape("int"), 40090448}
    # ){dshape("int64"), 40077744}

    if DEBUG:
        pprint(plan, width=1)

def test_complex_pipeline2():
    a = NDArray([1,2,3])
    b = NDArray([1,2,3])
    c = NDArray([1,2,3])
    d = NDArray([1,2,3])

    f = ((a*b)+(c*d))
    g = f**(a+b)

    line = Pipeline()
    _, plan = line.run_pipeline(f+g)

    #   Arithmetic(
    #     Add()
    #   , Arithmetic(
    #       Add()
    #     , Arithmetic(
    #         Mul()
    #       , Array(){dshape("3, int64"), 61582152}
    #       , Array(){dshape("3, int64"), 61469976}
    #       ){dshape("int64"), 61526768}
    #     , Arithmetic(
    #         Mul()
    #       , Array(){dshape("3, int64"), 61524520}
    #       , Array(){dshape("3, int64"), 61531056}
    #       ){dshape("int64"), 61526864}
    #     ){dshape("int64"), 61527152}
    #   , Pow(
    #       Arithmetic(
    #         Add()
    #       , Arithmetic(
    #           Mul()
    #         , Array(){dshape("3, int64"), 61582152}
    #         , Array(){dshape("3, int64"), 61469976}
    #         ){dshape("int64"), 61526768}
    #       , Arithmetic(
    #           Mul()
    #         , Array(){dshape("3, int64"), 61524520}
    #         , Array(){dshape("3, int64"), 61531056}
    #         ){dshape("int64"), 61526864}
    #       ){dshape("int64"), 61527152}
    #     , Arithmetic(
    #         Add()
    #       , Array(){dshape("3, int64"), 61582152}
    #       , Array(){dshape("3, int64"), 61469976}
    #       ){dshape("int64"), 61528304}
    #     ){dshape("int64"), 61528496}
    #   ){dshape("int64"), 61528592}

    if DEBUG:
        pprint(plan, width=1)

if __name__ == '__main__':
    test_simple_pipeline()
