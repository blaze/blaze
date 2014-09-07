from blaze.expr.array import *
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


def test_element_record():
    x = ArraySymbol('x', '5 * 3 * {name: string, amount: float32}')
    assert isinstance(x[1, 2], Scalar)
    assert x[1, 2].dshape == dshape('{name: string, amount: float32}')
    assert x[1, 2].names == x.names


def test_slice():
    x = ArraySymbol('x', '5 * 3 * {name: string, amount: float32}')
    assert x[2:, 0].dshape == dshape('3 * {name: string, amount: float32}')

    assert x[2:].dshape == x[2:, :].dshape


def test_reduction_dshape():
    x = ArraySymbol('x', '5 * 3 * float32')
    assert x.sum().dshape == x.schema
    assert x.sum(axis=0).dshape == dshape('3 * float32')
    assert x.sum(axis=1).dshape == dshape('5 * float32')
    assert x.sum(axis=(0, 1)).dshape == dshape('float32')
