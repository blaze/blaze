from blaze.data.dynd import *

from dynd import nd

from unittest import TestCase

class TestDyND(TestCase):

    data = [[1, 1], [2, 2]]
    def setUp(self):
        arr = nd.array(self.data, dtype='2 * 2 * int32')
        self.dd = DyND(arr)

    def test_dshape(self):
        assert str(self.dd.dshape) == '2 * 2 * int32'
        assert str(self.dd.schema) == '2 * int32'

    def test_iteration(self):
        assert tuple(map(tuple, self.dd)) == ((1, 1), (2, 2))

    def test_chunks(self):
        chunks = list(self.dd.chunks())

        assert all(isinstance(chunk, nd.array) for chunk in chunks)
        assert nd.as_py(chunks[0]) == self.data

    def test_as_dynd(self):
        assert isinstance(self.dd.as_dynd(), nd.array)

    def test_indexing(self):
        assert self.dd[0, 0] == 1
        assert tuple(self.dd[:, 0]) == (1, 2)
