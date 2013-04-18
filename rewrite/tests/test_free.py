from rewrite import aparse
from rewrite import terms as ast
from rewrite.dsl import dslparse
from rewrite.matching import free, NoMatch
from rewrite.dsl.toplevel import build_rule

from nose.tools import assert_raises

def test_linear():
    x = aparse('x')
    y = aparse('y')

    a0 = aparse('f(x,y)')
    v = free(a0)

    assert list(v) == [
        (x, 'x', ast.aterm),
        (y, 'y', ast.aterm)
    ]

def test_nonlinear():
    x = aparse('x')
    y = aparse('y')

    a0 = aparse('f(x,x,y)')
    v = free(a0)

    assert list(v) == [
        (x, 'x', ast.aterm),
        (x, 'x', ast.aterm),
        (y, 'y', ast.aterm)
    ]

def test_aspattern1():
    x = aparse('x')

    a0 = dslparse('b: f(a@x) -> a')
    v = free(a0[0].lhs)

    assert list(v) == [
        (x, 'a', ast.aterm)
    ]

def test_aspattern2():
    a0 = dslparse('b: @f(x,y) -> a')
    v = free(a0[0].lhs)

    f = aparse('f')
    x = aparse('x')
    y = aparse('y')

    assert list(v) == [
        (f, 'f', ast.aterm),
        (x, 'x', ast.aterm),
        (y, 'y', ast.aterm)
    ]

def test_aspattern3():
    a0 = dslparse('b: a@f(x,y) -> a')
    v = free(a0[0].lhs)

    f = aparse('f(x,y)')

    assert list(v) == [
        (f, 'a', ast.aappl)
    ]

# --------

def test_rule1():
    rule = dslparse('b: f(x,y) -> f(x,x,y,x,f(x,y))')
    l = rule[0].lhs
    r = rule[0].rhs

    rr = build_rule(l,r)

    sub = aparse('f(1,2)')
    res = aparse('f(1,1,2,1,f(1,2))')
    assert rr(sub) == res

def test_rule2():
    rule = dslparse('b: a@f(x,y) -> g(a,a(a))')
    l = rule[0].lhs
    r = rule[0].rhs

    rr = build_rule(l,r)

    sub = aparse('f(1,2)')
    res = aparse('g(f(1, 2), a(f(1, 2)))')
    assert rr(sub) == res

def test_rule3():
    rule = dslparse('b: @a(x,y) -> F(a,y,x)')
    l = rule[0].lhs
    r = rule[0].rhs

    rr = build_rule(l,r)

    sub = aparse('f(1,2)')
    res = aparse('F(f,2,1)')
    assert rr(sub) == res

def test_rule4():
    rule = dslparse('b: f(x,x) -> x')
    l = rule[0].lhs
    r = rule[0].rhs

    rr = build_rule(l,r)

    sub = aparse('f(1,1)')
    res = aparse('1')
    assert rr(sub) == res

    with assert_raises(NoMatch):
        sub = aparse('f(1,2)')
        assert rr(sub) == res

def test_rule5():
    rule = dslparse('b: f(x,g(y,z)) -> (x,y,z)')
    l = rule[0].lhs
    r = rule[0].rhs

    rr = build_rule(l,r)

    sub = aparse('f(1,g(2,3))')
    res = aparse('(1,2,3)')
    assert rr(sub) == res
