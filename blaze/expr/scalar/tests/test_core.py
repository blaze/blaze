from blaze.expr.scalar.core import ScalarSymbol, eval_str
from datashape import dshape


def test_basic():
    x = ScalarSymbol('x', 'real')
    assert eval(str(x)) == x

    assert x.dshape == dshape('real')


def test_eval_str():
    assert eval_str(1) == '1'
    assert eval_str('Alice') == "'Alice'"
