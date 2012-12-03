from numpy import array, arange, dtype
from blaze.slicealgebra import numpy_get

def test_contains_numpy1D():
    na = array([1,2,3])

    for i, el in enumerate(iter(na)):
        x = numpy_get(na, (i,))
        assert x == el

def test_contains_numpy2D():
    na = array([[1,2,3], [4,5,6]])

    # there are better ways to do this, just testing the
    # correspondance
    for i, a in enumerate(iter(na)):
        for j, b in enumerate(iter(a)):
            x = numpy_get(na, (i,j))
            assert x == b

def test_contains_numpy3D():
    na = arange(3*3*3).reshape(3,3,3)

    for i, a in enumerate(iter(na)):
        for j, b in enumerate(iter(a)):
            for k, c in enumerate(iter(b)):
                x = numpy_get(na, (i,j,k))
                assert x == c

def test_contains_float():
    na = array([1,2,3], dtype('f8'))

    for i, el in enumerate(iter(na)):
        x = numpy_get(na, (i,))
        assert x == el

    na = arange(3*3*3).reshape(3,3,3)
    na = na.astype('float')

def test_contains_bool():
    na = array([True, False, True])

    for i, el in enumerate(iter(na)):
        x = numpy_get(na, (i,))
        assert x == el
