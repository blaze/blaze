from blaze.expr.utils import *
from blaze.expr.utils import _slice


def test__slice_object():
    s = _slice(1, 10, 2)
    assert str(s) == '1:10:2'
    assert hash(s)

    assert _slice(1, 10, 2) == _slice(1, 10, 2)
