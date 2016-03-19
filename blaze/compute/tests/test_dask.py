from __future__ import absolute_import, division, print_function

import pytest
import sys

dask = pytest.importorskip('dask')

import numpy as np

from dask.array import Array, from_array
from odo import discover
from operator import getitem

from blaze import compute, symbol


def eq(a, b):
    if isinstance(a, Array):
        a = a.compute()
    if isinstance(b, Array):
        b = b.compute()
    c = a == b
    if isinstance(c, np.ndarray):
        c = c.all()
    return c


nx = np.arange(600).reshape((20, 30))
dx = from_array(nx, (4, 5))
sx = symbol('x', discover(dx))

ny = np.arange(600).reshape((30, 20))
dy = from_array(ny, (5, 4))
sy = symbol('y', discover(dy))

na = np.arange(20)
da = from_array(na, (4,))
sa = symbol('a', discover(da))

nb = np.arange(30).reshape((30, 1))
db = from_array(nb, (5, 1))
sb = symbol('b', discover(db))

dask_ns = {sx: dx, sy: dy, sa: da, sb: db}
numpy_ns = {sx: nx, sy: ny, sa: na, sb: nb}

exprs = [2 * sx + 1,
         sx.sum(axis=0),
         pytest.mark.skipif(True, sx.mean(axis=0),
                            reason="Segfaulting and remote process error"),
         sx + sx,
         sx.T,
         sx.T + sy,
         sx.dot(sy),
         sy.dot(sx),
         sx.sum(),
         sx - sx.sum(),
         sx.dot(sx.T),
         sx.sum(axis=1),
         sy + sa,
         sy + sb,
         sx[3:17],
         sx[3:10, 10:25:2] + 1,
         sx[:5, 10],
         sx[0, 0]]


@pytest.mark.parametrize('expr', exprs)
@pytest.mark.skipif(sys.platform == 'win32' and sys.version_info[:2] == (2, 7),
		    reason="Nondeterministic exceptions relating to get_async")
def test_compute(expr):
    result = compute(expr, dask_ns)
    expected = compute(expr, numpy_ns)
    assert isinstance(result, Array)
    if expr.dshape.shape:
        result2 = np.array(result)
    else:
        result2 = float(result)
    assert eq(result2, expected)


def test_elemwise_broadcasting():
    arr = compute(sy + sb, dask_ns)
    expected = [[(arr.name, i, j) for j in range(5)]
                for i in range(6)]
    assert arr._keys() == expected


def test_ragged_blockdims():
    dsk = {('x', 0, 0): np.ones((2, 2)),
           ('x', 0, 1): np.ones((2, 3)),
           ('x', 1, 0): np.ones((5, 2)),
           ('x', 1, 1): np.ones((5, 3))}

    a = Array(dsk, 'x', chunks=[(2, 5), (2, 3)], shape=(7, 5))
    s = symbol('s', '7 * 5 * int')

    assert compute(s.sum(axis=0), a).chunks == ((2, 3),)
    assert compute(s.sum(axis=1), a).chunks == ((2, 5),)

    assert compute(s + 1, a).chunks == a.chunks


def test_slicing_with_singleton_dimensions():
    arr = compute(sx[5:15, 12], dx)
    x = dx.name
    y = arr.name
    assert arr.dask[(y, 0)] == (getitem, (x, 1, 2), (slice(1, 4, 1), 2))
    assert arr.dask[(y, 1)] == (
        getitem, (x, 2, 2), (slice(None, None, None), 2))
    assert arr.dask[(y, 2)] == (getitem, (x, 3, 2), (slice(0, 3, 1), 2))
    assert all(len(k) == 2 for k in arr._keys())


def test_slicing_with_lists():
    nx = np.arange(20).reshape((4, 5))
    dx = from_array(nx, (2, 2))
    sx = symbol('x', discover(dx))

    expr = sx[[2, 0, 3]]
    assert eq(np.array(compute(expr, dx)), compute(expr, nx))

    expr = sx[::2, [2, 0, 3]]
    assert eq(np.array(compute(expr, dx)), compute(expr, nx))

    expr = sx[1, [2, 0, 3]]
    assert eq(np.array(compute(expr, dx)), compute(expr, nx))

    expr = sx[[2, 0, 3], -2]
    assert eq(np.array(compute(expr, dx)), compute(expr, nx))

    expr = sx[:, :]
    assert compute(expr, dx).dask == dx.dask

    expr = sx[0]
    assert eq(np.array(compute(expr, dx)), compute(expr, nx))

    expr = sx[0, [3, 1, 4]]
    assert eq(np.array(compute(expr, dx)), compute(expr, nx))


def test_slicing_on_boundary_lines():
    nx = np.arange(100).reshape((10, 10))
    dx = from_array(nx, (3, 3))
    sx = symbol('x', discover(dx))
    expr = sx[0, [1, 3, 9, 3]]

    result = compute(expr, dx)
    assert eq(result, nx[0, [1, 3, 9, 3]])


def test_slicing_with_newaxis():
    nx = np.arange(20).reshape((4, 5))
    dx = from_array(nx, (2, 2))
    sx = symbol('x', discover(dx))

    expr = sx[:, None, :]
    result = compute(expr, dx)

    assert result.shape == (4, 1, 5)
    assert result.chunks == ((2, 2), (1,), (2, 2, 1))
    assert eq(np.array(result), compute(expr, nx))

    expr = sx[None, [2, 1, 3], None, None, :, None]
    result = compute(expr, dx)

    assert result.shape == (1, 3, 1, 1, 5, 1)
    assert result.chunks == ((1,), (3,), (1,), (1,), (2, 2, 1), (1,))
    assert eq(np.array(result), compute(expr, nx))
