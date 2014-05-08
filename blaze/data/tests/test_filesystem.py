import os
from contextlib import contextmanager
from unittest import TestCase
from dynd import nd

from blaze.data import Concat, CSV, Stack
from blaze.utils import filetexts




class Test_Files(TestCase):
    data = {'a.csv': '1,1\n2,2',
            'b.csv': '3,3\n4,4\n5,5',
            'c.csv': '6,6\n7,7'}
    def test_Concat(self):
        with filetexts(self.data) as filenames:
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

class Test_Stack(TestCase):
    data = {'a.csv': '1,1\n2,2',
            'b.csv': '3,3\n4,4',
            'c.csv': '5,5\n6,6'}
    def test_Stack(self):
        with filetexts(self.data) as filenames:
            descriptors = [CSV(fn, schema='2 * int32')
                            for fn in sorted(filenames)]
            dd = Stack(descriptors)
            self.assertEqual(dd.dshape, 3 * descriptors[0].dshape)

            expected = (((1, 1), (2, 2)),
                        ((3, 3), (4, 4)),
                        ((5, 5), (6, 6)))

            self.assertEqual(dd.as_py(), expected)

            result = dd.as_dynd()
            expected2 = nd.array(expected, dtype='int32')
            self.assertEqual(nd.as_py(result),
                             nd.as_py(expected2))

            self.assertEqual(tuple(dd), expected)
            self.assertEqual(tuple(dd), expected)  # Not one use only

            chunks = dd.chunks()
            assert all(isinstance(chunk, nd.array) for chunk in chunks)
