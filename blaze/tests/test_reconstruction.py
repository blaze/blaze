from blaze.reconstruction import *

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
        "?"       : dynamic_t,
        "product" : Function(var1, Function(var2, product_t)),
        "sum"     : Function(var1, Function(var2, sum_t)),
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
