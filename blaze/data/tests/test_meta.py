from __future__ import absolute_import, division, print_function

from collections import Iterator
from dynd import nd
import json
import pytest

from blaze.data import Concat, CSV, Stack, JSON_Streaming
from blaze.utils import filetexts
from blaze.data.utils import tuplify


@pytest.yield_fixture
def file_data():
    data = {'a.csv': '1,1\n2,2',
            'b.csv': '3,3\n4,4\n5,5',
            'c.csv': '6,6\n7,7'}
    with filetexts(data) as filenames:
        descriptors = [CSV(fn, schema='2 * int32')
                       for fn in sorted(filenames)]
        yield Concat(descriptors)


def test_concat(file_data):
    dd = file_data
    assert str(dd.schema) == '2 * int32'
    assert str(dd.dshape) == 'var * 2 * int32'

    expected = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7))

    assert tuplify(tuple(dd)) == expected

    result = dd.as_dynd()
    expected2 = nd.array(expected, dtype='int32')
    nd.as_py(result) == nd.as_py(expected2)

    assert tuplify(tuple(dd)) == expected
    assert tuplify(tuple(dd)) == expected  # Not one use only

    chunks = list(dd.chunks())
    assert all(isinstance(chunk, nd.array) for chunk in chunks)

    tuple(dd[[0, 2], 0]) == (1, 3)
    tuple(dd[2, [1, 0]]) == (3, 3)

    assert isinstance(dd[:, 0], Iterator)


@pytest.yield_fixture
def stack_data():
    data = {'a.csv': '1,1\n2,2',
            'b.csv': '3,3\n4,4',
            'c.csv': '5,5\n6,6'}
    with filetexts(data) as filenames:
        yield filenames


def test_stack(stack_data):
    descriptors = [CSV(fn, schema='2 * int32') for fn in sorted(stack_data)]
    dd = Stack(descriptors)
    assert dd.dshape == 3 * descriptors[0].dshape

    expected = (((1, 1), (2, 2)),
                ((3, 3), (4, 4)),
                ((5, 5), (6, 6)))

    assert tuplify(tuple(dd.as_py())) == expected

    result = dd.as_dynd()
    expected2 = nd.array(expected, dtype='int32')
    assert nd.as_py(result) == nd.as_py(expected2)

    assert tuplify(tuple(dd)) == expected
    assert tuplify(tuple(dd)) == expected  # Not one use only

    chunks = dd.chunks()
    assert all(isinstance(chunk, nd.array) for chunk in chunks)

    assert tuple(dd[[0, 2], 0, 0]) == (1, 5)
    assert tuplify(tuple(dd[0])) == ((1, 1), (2, 2))
    res = dd[0, :, [1]]
    x = tuple(res)
    assert tuplify(x) == ((1,), (2,))
    assert tuplify(tuple(dd[0])) == expected[0]

    assert isinstance(dd[:, 0], Iterator)
    assert isinstance(dd[:], Iterator)


@pytest.yield_fixture
def json_data():
    data = {'a.csv': [{'x':  1, 'y':  2}, {'x':  3, 'y':  4}],
            'b.csv': [{'x':  5, 'y':  6}, {'x':  7, 'y':  8}],
            'c.csv': [{'x':  9, 'y': 10}, {'x': 11, 'y': 12}]}

    text = dict((fn, '\n'.join(map(json.dumps, dicts)))
                for fn, dicts in data.items())
    with filetexts(text) as filenames:
        descriptors = [JSON_Streaming(fn, schema='{x: int32, y: int32}')
                       for fn in sorted(filenames)]
        yield Stack(descriptors)


def test_stack(json_data):
    dd = json_data

    expected = (((1, 2), (3, 4)),
                ((5, 6), (7, 8)),
                ((9, 10), (11, 12)))

    assert tuplify(dd.as_py()) == expected

    tuplify(dd[::2, 1, :]) == ((3, 4), (11, 12))
    tuplify(dd[::2, 1, 'x']) == (3, 11)
