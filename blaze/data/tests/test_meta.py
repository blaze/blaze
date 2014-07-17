from __future__ import absolute_import, division, print_function

import os
from collections import Iterator
from contextlib import contextmanager
from unittest import TestCase
from dynd import nd
import json

from blaze.data import Concat, CSV, Stack, JSON_Streaming
from blaze.utils import filetexts
from blaze.data.utils import tuplify



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

            self.assertEqual(tuplify(tuple(dd)), expected)

            result = dd.as_dynd()
            expected2 = nd.array(expected, dtype='int32')
            self.assertEqual(nd.as_py(result),
                             nd.as_py(expected2))

            self.assertEqual(tuplify(tuple(dd)), expected)
            self.assertEqual(tuplify(tuple(dd)), expected)  # Not one use only

            chunks = list(dd.chunks())
            assert all(isinstance(chunk, nd.array) for chunk in chunks)

            self.assertEqual(tuple(dd[[0, 2], 0]), (1, 3))
            self.assertEqual(tuple(dd[2, [1, 0]]), (3, 3))

            assert isinstance(dd[:, 0], Iterator)


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

            self.assertEqual(tuplify(tuple(dd.as_py())), expected)

            result = dd.as_dynd()
            expected2 = nd.array(expected, dtype='int32')
            self.assertEqual(nd.as_py(result),
                             nd.as_py(expected2))

            self.assertEqual(tuplify(tuple(dd)), expected)
            self.assertEqual(tuplify(tuple(dd)), expected)  # Not one use only

            chunks = dd.chunks()
            assert all(isinstance(chunk, nd.array) for chunk in chunks)

            self.assertEqual(tuple(dd[[0, 2], 0, 0]), (1, 5))
            self.assertEqual(tuplify(tuple(dd[0])), ((1, 1), (2, 2)))
            self.assertEqual(tuplify(tuple(dd[0, :, [1]])), ((1,), (2,)))
            self.assertEqual(tuplify(tuple(dd[0])), expected[0])

            assert isinstance(dd[:, 0], Iterator)
            assert isinstance(dd[:], Iterator)



class Test_Stack_JSON(TestCase):
    data = {'a.csv': [{'x':  1, 'y':  2}, {'x':  3, 'y':  4}],
            'b.csv': [{'x':  5, 'y':  6}, {'x':  7, 'y':  8}],
            'c.csv': [{'x':  9, 'y': 10}, {'x': 11, 'y': 12}]}

    text = dict((fn, '\n'.join(map(json.dumps, dicts)))
                    for fn, dicts in data.items())
    def test_Stack(self):
        with filetexts(self.text) as filenames:
            descriptors = [JSON_Streaming(fn, schema='{x: int32, y: int32}')
                            for fn in sorted(filenames)]
            dd = Stack(descriptors)

            expected = (((1,  2), ( 3,  4)),
                        ((5,  6), ( 7,  8)),
                        ((9, 10), (11, 12)))

            self.assertEqual(tuplify(dd.as_py()), expected)

            self.assertEqual(tuplify(dd[::2, 1, :]), ((3, 4), (11, 12)))
            self.assertEqual(tuplify(dd[::2, 1, 'x']), (3, 11))
