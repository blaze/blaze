from blaze.aterm import parse
from blaze.rts.funcs import lookup

from blaze import NDArray, Array
from blaze import add, multiply

from unittest import skip

def test_match1():
    expr = parse('Add(1,2)')
    fn, cost = lookup(expr)
    #assert fn.fn == add.fn.im_func

def test_match2():
    expr = parse('Mul(1,2)')
    fn, cost = lookup(expr)
    #assert fn.fn == multiply.fn.im_func

@skip
def test_manifest_func():
    x = Array([1,2,3])
    y = Array([1,2,3])

    val = add(x,y)

    assert val[0] == 2
    assert val[1] == 4
    assert val[2] == 6

def test_deferred_func():
    x = NDArray([1,2,3])
    y = NDArray([1,2,3])

    val = add(x,y)
