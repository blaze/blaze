from blaze.expr import Expr, symbol, Cast
from blaze.compute import compute_up
from blaze import compute


class Foo(object):
    def __init__(self, data):
        self.data = data


@compute_up.register((Expr, Cast), Foo)
def compute_up(expr, data, **kwargs):
    return data


def optimize(expr, data):
    """ Renames leaf """
    leaf = expr._leaves()[0]
    return expr._subs({leaf: symbol('newname', leaf.dshape)})


def test_scope_gets_updated_after_optimize_call():
    a = symbol('a', 'int')
    result = compute(a + 1, Foo('foo'), optimize=optimize)
    assert result.data == 'foo'
