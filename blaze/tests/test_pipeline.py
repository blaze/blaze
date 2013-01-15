# TODO assertsion for the output'd values... right now they have
# pointer addresses so its hard to do string matching

from blaze.expr import ops
from blaze.table import NDArray
from blaze.compile.pipeline import compile
from blaze.expr.graph import IntNode, FloatNode, VAL, OP, APP
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

def test_simple_pipeline():
    plan = compile(x)

    # Add(1,Mul(2,Abs(3.0)))
    if DEBUG:
        pprint(plan, width=1)

def test_simple_pipeline2():

    plan = compile(y)

    # Add(Add(1,Add(2,3.0)),Add(1,Mul(2,Abs(3.0))))
    if DEBUG:
        pprint(plan, width=1)

    plan = compile(x+y)

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

    plan = compile(x*(y+2)+3)

    #------------------------------------------------------------------------

    # Add(Array(39558864){dshape("3 int64")},Array(39558864){dshape("3 int64")})
    if DEBUG:
        pprint(plan, width=1)

    plan = compile(d+d)

    # Add(
    #   Add(Array(39558864){dshape("3 int64")}, Array(39558864){dshape("3 int64")})
    # , Add(Array(39558864){dshape("3 int64")}, Array(39558864){dshape("3 int64")})
    # )
    if DEBUG:
        pprint(plan, width=1)

    plan = compile((d+d)+(d+d))

    if DEBUG:
        pprint(plan, width=1)


    # Mul(
    #   Mul(Array(39558864){dshape("3 int64")}, Array(39558864){dshape("3 int64")})
    # , Mul(Array(39558864){dshape("3 int64")}, Array(39558864){dshape("3 int64")})
    # )
    plan = compile((d*d)*(d*d))

    if DEBUG:
        pprint(plan, width=1)

def test_complex_pipeline1():
    a = NDArray([1,2,3])
    b = NDArray([1,2,3])
    c = NDArray([1,2,3])
    d = NDArray([1,2,3])

    plan = compile(((a*b)+(c*d))**2)

    if DEBUG:
        pprint(plan, width=1)

def test_complex_pipeline2():
    a = NDArray([1,2,3])
    b = NDArray([1,2,3])
    c = NDArray([1,2,3])
    d = NDArray([1,2,3])

    f = ((a*b)+(c*d))
    g = f**(a+b)

    plan = compile(f+g)

    if DEBUG:
        pprint(plan, width=1)

if __name__ == '__main__':
    test_simple_pipeline()
