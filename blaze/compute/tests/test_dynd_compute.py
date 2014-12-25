from __future__ import absolute_import, division, print_function

import pytest
dynd = pytest.importorskip('dynd')
from dynd import nd

from blaze.compute.core import compute
from blaze.expr import *
from blaze.compute.dynd import *


def eq(a, b):
    return nd.as_py(a) == nd.as_py(b)


n = symbol('n', '3 * 5 * int')

nx = nd.array([[ 1,  2,  3,  4,  5],
               [11, 22, 33, 44, 55],
               [21, 22, 23, 24, 25]], type=str(n.dshape))

rec = symbol('s', '3 * var * {name: string, amount: int}')

recx = nd.array([[('Alice', 1), ('Bob', 2)],
                 [('Charlie', 3)],
                 [('Dennis', 4), ('Edith', 5), ('Frank', 6)]],
                type=str(rec.dshape))


def test_symbol():
    assert eq(compute(n, nx), nx)


def test_slice():
    assert eq(compute(n[0], nx), nx[0])
    assert eq(compute(n[0, :3], nx), nx[0, :3])


def test_first_last():
    assert eq(compute(n[0], nx), nx[0])
    assert eq(compute(n[-1], nx), nx[-1])


def test_field():
    assert eq(compute(rec.amount, recx), recx.amount)


def test_arithmetic():
    # assert eq(compute(n + 1, nx), nx + 1)
    assert eq(compute(rec.amount + 1, recx), recx.amount + 1)
    assert eq(compute(-rec.amount, recx), 0-recx.amount)
