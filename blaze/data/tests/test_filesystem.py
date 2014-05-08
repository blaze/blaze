import os
from contextlib import contextmanager
from unittest import TestCase
from dynd import nd

from blaze.data import Concat, CSV
from blaze.utils import filetexts


data = {'a.csv': '1,1\n2,2',
        'b.csv': '3,3\n4,4\n5,5',
        'c.csv': '6,6\n7,7'}


class Test_Files(TestCase):
    def test_filesystem(self):
        with filetexts(data) as filenames:
            descriptors = [CSV(fn, schema='2 * int32')
                            for fn in sorted(filenames)]
            dd = Concat(descriptors)

            self.assertEqual(str(dd.schema), '2 * int32')
            self.assertEqual(str(dd.dshape), 'var * 2 * int32')

            expected = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7))

            self.assertEqual(tuple(dd), expected)

            result = dd.as_dynd()
            expected2 = nd.array(expected, dtype='int32')
            self.assertEqual(nd.as_py(result),
                             nd.as_py(expected2))

            self.assertEqual(tuple(dd), expected)
            self.assertEqual(tuple(dd), expected)  # Not one use only

            chunks = list(dd.chunks())
            assert all(isinstance(chunk, nd.array) for chunk in chunks)
