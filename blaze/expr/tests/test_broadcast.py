from blaze.expr import *
from blaze.expr.broadcast2 import *

x = Symbol('x', '5 * 3 * int32')
xx = Symbol('xx', 'int32')

y = Symbol('y', '5 * 3 * int32')
yy = Symbol('yy', 'int32')

a = Symbol('a', 'int32')


def test_broadcast_basic():
    b = Broadcast((x, y), (xx, yy), xx + yy)

    assert b.shape == x.shape
    assert b.schema == (xx + yy).dshape

    assert eval(str(b)).isidentical(b)
