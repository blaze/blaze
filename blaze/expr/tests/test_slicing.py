from blaze.expr import symbol
import numpy as np
from datashape import dshape, isscalar


def test_array_dshape():
    x = symbol('x', '5 * 3 * float32')
    assert x.shape == (5, 3)
    assert x.schema == dshape('float32')
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


def test_negative_slice():
    x = symbol('x', '10 * 10 * int32')
    assert x[:5, -3:].shape == (5, 3)


def test_None_slice():
    x = symbol('x', '10 * 10 * int32')
    assert x[:5, None, -3:].shape == (5, 1, 3)


def test_list_slice():
    x = symbol('x', '10 * 10 * int32')
    assert x[[1, 2, 3], [4, 5]].shape == (3, 2)


def test_list_slice_string():
    x = symbol('x', '10 * 10 * int32')
    assert str(x[[1, 2, 3]]) == "x[[1, 2, 3]]"


def test_slice_with_boolean_list():
    x = symbol('x', '5 * int32')
    expr = x[[True, False, False, True, False]]
    assert expr.index == ([0, 3],)


def test_slice_with_numpy_array():
    x = symbol('x', '2 * int32')
    assert x[np.array([True, False])].isidentical(x[[True, False]])
