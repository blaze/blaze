import numpy as np

from ndtable.expr.ops import array_like
from ndtable.expr.paterm import ATerm, AAppl, AInt
from ndtable.rts.ffi import install, lift

def test_lift():

    # Simple function takes two array-like arguments yields an
    # array-like result.
    @lift('x -> x -> x', {'x': array_like})
    def nadd(a,b,o):
        return np.add(a,b,o)

def test_install():

    @lift('x -> x -> x', {'x': array_like})
    def nadd(a,b,o):
        return np.add(a,b,o)

    costfn = lambda x: 0

    # Anywhere an add is term is found replace with the simple NumPy
    # dispatch.
    install('Add(a,b);*',nadd,costfn)

    from ndtable.rts.ffi import _dispatch

    expr = AAppl(ATerm('Add'), [AInt(1), AInt(2)])
    fn, cost = _dispatch.dispatcher.dispatch(expr)

    assert fn == nadd
    assert cost == 0
