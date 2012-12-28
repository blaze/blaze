from blaze.aterm import parse
from blaze.rts.funcs import lookup

from blaze import add, multiply

def test_match1():
    expr = parse('Add(1,2)')
    fn, cost = lookup(expr)
    assert fn.fn == add.fn.im_func

def test_match2():
    expr = parse('Mul(1,2)')
    fn, cost = lookup(expr)
    assert fn.fn == multiply.fn.im_func
