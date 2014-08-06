from blaze.data.python import *
from blaze.data.core import discover
from blaze.data.utils import tuplify
from dynd import nd

def test_basic():
    data = ((1, 1), (2, 2))
    dd = Python([], schema='2 * int32')

    dd.extend(data)

    assert str(dd.dshape) == 'var * 2 * int32'
    assert str(dd.schema) == '2 * int32'

    assert tuplify(tuple(dd)) == data
    print(dd.as_py())
    assert dd.as_py() == data

    chunks = list(dd.chunks())

    assert all(isinstance(chunk, nd.array) for chunk in chunks)
    assert nd.as_py(chunks[0]) == list(map(list, data))

    assert isinstance(dd.as_dynd(), nd.array)

    assert tuple(dd[0]) == data[0]
    assert dd[0, 1] == data[0][1]
    assert tuple(dd[[0, 1], 1]) == (1, 2)


def test_discover():
    data = ((1, 1), (2, 2))
    dd = Python([], schema='2 * int32')
    assert discover(dd) == dd.dshape
