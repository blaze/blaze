from blaze.data.dynd import *

from dynd import nd

def test_basic():
    data = [[1, 1], [2, 2]]
    arr = nd.array(data, dtype='2 * 2 * int32')

    dd = DyND(arr)

    assert str(dd.dshape) == '2 * 2 * int32'
    assert str(dd.schema) == '2 * int32'

    assert list(dd) == [[1, 1], [2, 2]]
    chunks = list(dd.chunks())

    assert all(isinstance(chunk, nd.array) for chunk in chunks)
    assert nd.as_py(chunks[0]) == data

    assert isinstance(dd.dynd_arr(), nd.array)
