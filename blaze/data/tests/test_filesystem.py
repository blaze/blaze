import os
from contextlib import contextmanager
from unittest import TestCase
from dynd import nd

from blaze.data import Files, CSV
from blaze.utils import filetexts


data = {'a.csv': '1,1\n2,2',
        'b.csv': '3,3\n4,4\n5,5',
        'c.csv': '6,6\n7,7'}


class Test_Files(TestCase):
    def test_filesystem(self):
        with filetexts(data) as filenames:
            dd = Files(sorted(filenames), CSV, schema='2 * int32')

            self.assertEquals(dd.filenames, ['a.csv', 'b.csv', 'c.csv'])
            self.assertEqual(str(dd.schema), '2 * int32')

            expected = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]]

            self.assertEqual(list(dd), expected)
            self.assertEqual(list(dd), expected)  # Not one use only

            chunks = list(dd.chunks(blen=3))
            expected = [nd.array([[1, 1], [2, 2], [3, 3]], dtype='int32'),
                        nd.array([[4, 4], [5, 5], [6, 6]], dtype='int32')]

            assert all(nd.as_py(a) == nd.as_py(b) for a, b in zip(chunks, expected))
