from blaze.reconstruction import *

from nose.tools import assert_raises

DEBUG = True

def test_reconstruct():
    var1 = TypeVar()
    var2 = TypeVar()

    product_t = TypeCon("x", (var1, var2), infix=True)
    sum_t     = TypeCon("+", (var1, var2), infix=True)
    dynamic_t = TypeCon("?", [])

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

    x = lam(['x', 'y'], product(Atom('x'), Atom('y')))
    inferred = infer(env,x, debug=DEBUG)
    assert pprint(inferred) == '(a -> (b -> (b x a)))'

    # -- Example 2 --

    x = app(
        lam(['x', 'y', 'z'], product(Atom('x'), Atom('z'))),
        [Atom('1')]
    )
    inferred = infer(env,x, debug=DEBUG)
    assert pprint(inferred) == '(a -> (b -> (b x int)))'

    # -- Example 3 --

    x = app(
        lam(['x', 'y', 'z'], product(Atom('x'), Atom('z'))),
        [Atom('1'), Atom('2'), Atom('3')]
    )
    inferred = infer(env,x, debug=DEBUG)
    assert pprint(inferred) == '(int x int)'

    # -- Example 4 --

    x = app(Atom("product"), [Atom("?"), Atom("1")])
    inferred = infer(env, x, debug=DEBUG)
    assert pprint(inferred) == '(? x int)'

    # -- Example 5 --

    x = app(Atom("sum"), [Atom("?"), Atom("1")])
    inferred = infer(env, x, debug=DEBUG)
    assert pprint(inferred) == '(? + int)'

    # -- Example 6 --

    x = app(Atom("boolfunc"), [Atom('1')])
    with assert_raises(TypeError):
        infer(env, x, debug=DEBUG)

def test_simple_lisp_like():
    # Simple typed cons list for LISP like things.

    var1 = TypeVar()
    var2 = TypeVar()
    list_t = TypeCon(":", (var1, var2))

    env = {
        "true"  : Bool,
        "false" : Bool,
        "zero"  : Integer,
        "one"   : Integer,

        "cons" : Function(var1, Function(var2, list_t)),
        "nil"  : list_t
    }

def test_simple_dtype_like():
    var1 = TypeVar()
    var2 = TypeVar()
    var2 = TypeVar()

    int_   = TypeCon("int", [])
    float_ = TypeCon("float", [])
    bool_  = TypeCon("bool", [])
    dynamic_t = TypeCon("?", [])

    # fun map        :: ((a -> b), A a) -> A b
    # fun reduce     :: (((a,a) -> b), A a) -> A b
    # fun accumulate :: (((a,a) -> b), A a) -> A b
    # fun zipwith    :: (((a,b) -> c), A a, A b) -> A c

    Array = TypeCon('Array', [var1])

    env = {
        "?"        : dynamic_t,
        "true"     : bool_,
        "false"    : bool_,
        "map"      : Function(Function(var1, var2), Function(Array, Array))
    }

    ufunc = lam(['x'], Atom('x'))
    x = app(Atom("map"), [ufunc])

    inferred = infer(env, x, debug=DEBUG)
