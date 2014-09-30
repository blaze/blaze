from blaze.expr.method_dispatch import select_functions, name, partial


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

def isnine(x):
    return True


methods = [(int, set([inc, dec])),
           (str, set([lower, upper])),
           (iseven, set([halve])),
           (9, set([isnine]))]


def test_select_functions():
    assert select_functions(methods, 3) == {'inc': inc, 'dec': dec}
    assert select_functions(methods, 4) == {'inc': inc, 'dec': dec, 'halve': halve}
    assert select_functions(methods, 'A') == {'lower': lower, 'upper': upper}
    assert select_functions(methods, 9) == {'inc': inc, 'dec': dec, 'isnine':
        isnine}


def test_name():
    assert name(inc) == 'inc'
    assert name(partial(inc)) == name(inc)
