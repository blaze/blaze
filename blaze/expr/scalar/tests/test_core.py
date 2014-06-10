from blaze.expr.scalar import ScalarSymbol
from datashape import dshape


def test_basic():
    x = ScalarSymbol('x', 'real')
    assert eval(str(x)) == x

    assert x.dshape == dshape('real')
