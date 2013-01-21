# -*- coding: utf-8 -*-

from blaze.type_reconstruction import *

from blaze.test_utils import assert_raises

DEBUG = True

def test_reconstruct():
    var1 = var()
    var2 = var()

    product_t = con("x", (var1, var2), infix=True)
    sum_t     = con("+", (var1, var2), infix=True)
    dynamic_t = con("?", [])

    # Example Env
    #------------

    env = {
        "?"        : dynamic_t,
        "product"  : Function(var1, Function(var2, product_t)),
        "sum"      : Function(var1, Function(var2, sum_t)),
        "true"     : Bool,
        "false"    : Bool,
        "boolfunc" : Function(Bool, Bool),
    }

    # -- Example 1 --

    x = lam(['x', 'y'], product(atom('x'), atom('y')))
    inferred = infer(env,x, debug=DEBUG)
    assert pprint(inferred) == '(a -> (b -> (b x a)))'

    # -- Example 2 --

    x = app(
        lam(['x', 'y', 'z'], product(atom('x'), atom('z'))),
        [atom('1')]
    )
    inferred = infer(env,x, debug=DEBUG)
    assert pprint(inferred) == '(a -> (b -> (b x int)))'

    # -- Example 3 --

    x = app(
        lam(['x', 'y', 'z'], product(atom('x'), atom('z'))),
        [atom('1'), atom('2'), atom('3')]
    )
    inferred = infer(env,x, debug=DEBUG)
    assert pprint(inferred) == '(int x int)'

    # -- Example 4 --

    x = app(atom("product"), [atom("?"), atom("1")])
    inferred = infer(env, x, debug=DEBUG)
    assert pprint(inferred) == '(? x int)'

    # -- Example 5 --

    x = app(atom("sum"), [atom("?"), atom("1")])
    inferred = infer(env, x, debug=DEBUG)
    assert pprint(inferred) == '(? + int)'

    # -- Example 6 --

    x = app(atom("boolfunc"), [atom('1')])
    with assert_raises(TypeError):
        infer(env, x, debug=DEBUG)

def test_simple_lisp_like():
    # Simple typed cons list for LISP like things.

    var1 = var()
    var2 = var()
    list_t = con(":", (var1, var2))

    env = {
        "true"  : Bool,
        "false" : Bool,
        "zero"  : Integer,
        "one"   : Integer,

        "cons" : Function(var1, Function(var2, list_t)),
        "nil"  : list_t
    }

def test_simple_dtype_like():
    var1 = var()
    var2 = var()
    var2 = var()

    int_   = con("int", [])
    float_ = con("float", [])
    bool_  = con("bool", [])
    dynamic_t = con("?", [])

    # fun map        :: ((a -> b), A a) -> A b
    # fun reduce     :: (((a,a) -> b), A a) -> A b
    # fun accumulate :: (((a,a) -> b), A a) -> A b
    # fun zipwith    :: (((a,b) -> c), A a, A b) -> A c

    Array = con('Array', [var1])

    env = {
        "?"        : dynamic_t,
        "true"     : bool_,
        "false"    : bool_,
        "map"      : Function(Function(var1, var2), Function(Array, Array))
    }

    ufunc = lam(['x'], atom('x'))
    x = app(atom("map"), [ufunc])

    inferred = infer(env, x, debug=DEBUG)
