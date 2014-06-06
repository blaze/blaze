from blaze.expr.scalar.core import ScalarSymbol
from datashape import dshape


def test_basic():
    s = ScalarSymbol('x', 'real')
    assert eval(str(s)) == s

    assert s.dshape == dshape('real')
