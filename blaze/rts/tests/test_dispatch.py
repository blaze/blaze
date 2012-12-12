import numpy as np

from blaze.expr.ops import array_like
from blaze.expr.paterm import ATerm, AAppl, AInt
from blaze.rts.ffi import install, lift, lookup

from unittest import skip

def test_install():

    expr = AAppl(ATerm('Add'), [AInt(1), AInt(2)])
    fn, cost = lookup(expr)

    assert fn.fn == np.add
    assert cost == 0

if __name__ == '__main__':
    test_install()
