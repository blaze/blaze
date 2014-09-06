from blaze.expr.array import *
import numpy as np

def test_dshape():
    x = ArraySymbol('x', '5 * 3 * float32')
    assert x.shape == (5, 3)
    assert x.dtype == np.float32
    assert x
    assert len(x) == 5

def test_dshape_record_type():
    x = ArraySymbol('x', '5 * 3 * {name: string, amount: float32}')
    assert x.shape == (5, 3)
    assert x.names == ['name', 'amount']
