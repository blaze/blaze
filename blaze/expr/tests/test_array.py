from blaze.expr.array import *
from blaze.expr.array import _slice
import numpy as np

def test_dshape():
    x = ArraySymbol('x', '5 * 3 * float32')
    assert x.shape == (5, 3)
    assert x.dtype == np.float32
    assert x.schema == dshape('float32')
    assert x
    assert len(x) == 5
    assert x.ndim == 2


def test_dshape_record_type():
    x = ArraySymbol('x', '5 * 3 * {name: string, amount: float32}')
    assert x.schema == dshape('{name: string, amount: float32}')
    assert x.shape == (5, 3)
    assert x.names == ['name', 'amount']
    assert x.ndim == 2


def test_element():
    x = ArraySymbol('x', '5 * 3 * float32')
    assert isinstance(x[1, 2], Scalar)
    assert x[1, 2].dshape == dshape('float32')

    x = ArraySymbol('x', '5 * float32')
    assert isinstance(x[3], Scalar)

    assert str(x[1, 2]) == 'x[1, 2]'


def test_element_record():
    x = ArraySymbol('x', '5 * 3 * {name: string, amount: float32}')
    assert isinstance(x[1, 2], Scalar)
    assert x[1, 2].dshape == dshape('{name: string, amount: float32}')
    assert x[1, 2].names == x.names


def test__slice_object():
    s = _slice(1, 10, 2)
    assert str(s) == '1:10:2'
    assert hash(s)

    assert _slice(1, 10, 2) == _slice(1, 10, 2)


def test_slice():
    x = ArraySymbol('x', '5 * 3 * {name: string, amount: float32}')
    assert x[2:, 0].dshape == dshape('3 * {name: string, amount: float32}')

    assert x[2:].dshape == x[2:, :].dshape

    assert isinstance(x[0, :2], Slice)
    assert isinstance(x[0, 2], Element)

    hash(x[:2])
    hash(x[0, :2])

    assert str(x[1]) == 'x[1]'
    assert str(x[:2]) == 'x[:2]'
    assert str(x[0, :2]) == 'x[0, :2]'
    assert str(x[1:4:2, :2]) == 'x[1:4:2, :2]'


def test_reduction_dshape():
    x = ArraySymbol('x', '5 * 3 * float32')
    assert x.sum().dshape == x.schema
    assert x.sum(axis=0).dshape == dshape('3 * float32')
    assert x.sum(axis=1).dshape == dshape('5 * float32')
    assert x.sum(axis=(0, 1)).dshape == dshape('float32')
