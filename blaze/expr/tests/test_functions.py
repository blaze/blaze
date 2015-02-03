from __future__ import absolute_import, division, print_function

from blaze.expr.functions import *
from blaze.expr import Expr, symbol
from blaze.expr import math
import numpy as np


def test_reductions():
    assert max([1, 2, 3]) == 3
    assert max(np.array([1, 2, 3])) == 3
    assert isinstance(max(symbol('t', 'var * {x: int, y: int}').x), Expr)
    assert all([True, True, True]) is True
    assert all([True, True, False]) is False


def test_math():
    assert sin(0) == 0
    assert isinstance(sin(0), float)
    assert isinstance(sin(np.int32(0)), np.float)
    assert isinstance(sin(symbol('x', 'real')), Expr)


def test_abs():
    assert abs(-1) == 1
    assert (abs(np.array([-1, 2, 3])) == np.array([1, 2, 3])).all()
    x = symbol('x', 'int')
    assert isinstance(abs(x), math.abs)
