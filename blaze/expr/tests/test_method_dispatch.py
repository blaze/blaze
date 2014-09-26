from blaze.expr.method_dispatch import get_methods


def iseven(n):
    return isinstance(n, int) and n % 2 == 0

def inc(x):
    return Foo(x.data + 1)

def dec(x):
    return Foo(x.data - 1)

def lower(x):
    return Foo(x.data.lower())

def upper(x):
    return Foo(x.data.upper())

def halve(x):
    return Foo(x.data // 2)


methods = [(int, {inc, dec}),
           (str, {lower, upper}),
           (iseven, {halve})]


def test_get_methods():
    assert get_methods(methods, 3) == {'inc': inc, 'dec': dec}
    assert get_methods(methods, 4) == {'inc': inc, 'dec': dec, 'halve': halve}
    assert get_methods(methods, 'A') == {'lower': lower, 'upper': upper}
