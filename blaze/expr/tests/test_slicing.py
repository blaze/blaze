from blaze.expr import symbol
from datashape import dshape, isscalar


def test_array_dshape():
    x = symbol('x', '5 * 3 * float32')
    assert x.shape == (5, 3)
    assert x.schema == dshape('float32')
    assert x
    assert len(x) == 5
    assert x.ndim == 2


def test_element():
    x = symbol('x', '5 * 3 * float32')
    assert isscalar(x[1, 2].dshape)
    assert x[1, 2].dshape == dshape('float32')

    assert str(x[1, 2]) == 'x[1, 2]'

    x = symbol('x', '5 * float32')
    assert isscalar(x[3].dshape)


def test_slice():
    x = symbol('x', '5 * 3 * {name: string, amount: float32}')
    assert x[2:, 0].dshape == dshape('3 * {name: string, amount: float32}')

    assert x[2:].dshape == x[2:, :].dshape

    # Make sure that these are hashable
    hash(x[:2])
    hash(x[0, :2])

    assert str(x[1]) == 'x[1]'
    assert str(x[:2]) == 'x[:2]'
    assert str(x[0, :2]) == 'x[0, :2]'
    assert str(x[1:4:2, :2]) == 'x[1:4:2, :2]'
