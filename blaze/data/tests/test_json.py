from __future__ import absolute_import, division, print_function

import unittest
import os
import tempfile
import json
from dynd import nd
import datashape
from collections import Iterator
from datashape.discovery import discover

from blaze.data import JSON, JSON_Streaming
from blaze.utils import filetext, raises
from blaze.data.utils import tuplify


class TestBigJSON(unittest.TestCase):
    maxDiff = None
    data = {
        "type": "ImageCollection",
        "images": [{
               "Width":  800,
                "Height": 600,
                "Title":  "View from 15th Floor",
                "Thumbnail": {
                    "Url":    "http://www.example.com/image/481989943",
                    "Height": 125,
                    "Width":  "100"
                },
                "IDs": [116, 943, 234, 38793]
            }]
    }

    ordered = (u'ImageCollection',
            ((800, 600, u'View from 15th Floor',
                (u'http://www.example.com/image/481989943', 125, 100),
                (116, 943, 234, 38793)),))

    dshape = """{
      type: string,
      images: var * {
            Width: int16,
            Height: int16,
            Title: string,
            Thumbnail: {
                Url: string,
                Height: int16,
                Width: int16,
            },
            IDs: var * int32,
        }
    }
    """

    def setUp(self):
        self.filename= tempfile.mktemp(".json")
        with open(self.filename, "w") as f:
            json.dump(self.data, f)

    def tearDown(self):
        os.remove(self.filename)

    def test_basic(self):
        dd = JSON(self.filename, 'r', dshape=self.dshape)
        self.assertRaises(Exception, lambda: tuple(dd))

    def test_as_py(self):
        dd = JSON(self.filename, 'r', dshape=self.dshape)
        self.assertEqual(tuplify(dd.as_py()), self.ordered)

    def test_discovery(self):
        dd = JSON(self.filename, 'r')
        s = str(dd.dshape)
        for word in ['Thumbnail', 'string', 'int', 'images', 'type']:
            assert word in s


json_buf = u"[1, 2, 3, 4, 5]"
json_dshape = "var * int8"


class TestJSON(unittest.TestCase):

    def setUp(self):
        handle, self.json_file = tempfile.mkstemp(".json")
        with os.fdopen(handle, "w") as f:
            f.write(json_buf)

    def tearDown(self):
        os.remove(self.json_file)

    def test_raise_error_on_non_existent_file(self):
        self.assertRaises(ValueError,
                    lambda: JSON('does-not-exist23424.josn', 'r'))

    def test_basic_object_type(self):
        dd = JSON(self.json_file, dshape=json_dshape)
        self.assertEqual(list(dd), [1, 2, 3, 4, 5])

    def test_iter(self):
        dd = JSON(self.json_file, dshape=json_dshape)
        # This equality does not work yet
        # self.assertEqual(dd.dshape, datashape.dshape(
        #     'Var, %s' % json_schema))
        self.assertEqual(list(dd), [1, 2, 3, 4, 5])

class AccountTestData(unittest.TestCase):
    def setUp(self):
        self.fn = tempfile.mktemp(".json")
        with open(self.fn, 'w') as f:
            for d in self.dicts:
                f.write(json.dumps(d))
                f.write('\n')
        self.dd = JSON_Streaming(self.fn, schema=self.schema)

    def tearDown(self):
        if os.path.exists(self.fn):
            os.remove(self.fn)

    dicts = [{'name': 'Alice', 'amount': 100},
             {'name': 'Alice', 'amount': 50},
             {'name': 'Bob', 'amount': 10},
             {'name': 'Charlie', 'amount': 200},
             {'name': 'Bob', 'amount': 100}]

    tuples = (('Alice', 100),
              ('Alice', 50),
              ('Bob', 10),
              ('Charlie', 200),
              ('Bob', 100))

    text = '\n'.join(map(json.dumps, dicts))

    schema = '{name: string, amount: int32}'


class TestDiscovery(AccountTestData):
    def test_discovery(self):
        dd = JSON_Streaming(self.fn)
        assert set(dd.schema[0].names) == set(['name', 'amount'])
        assert 'string' in str(dd.schema[0]['name'])


class Test_Indexing(AccountTestData):
    def test_indexing_basic(self):
        assert tuplify(self.dd[0]) == self.tuples[0]
        assert tuplify(self.dd[0:3]) == self.tuples[0:3]
        assert tuplify(self.dd[0::2]) == self.tuples[0::2]
        self.assertEqual(tuplify(self.dd[[3, 1, 3]]),
                         tuple(self.tuples[i] for i in [3, 1, 3]))

    def test_indexing_nested(self):
        assert tuplify(self.dd[0, 'name']) == self.tuples[0][0]
        assert tuplify(self.dd[0, 0]) == self.tuples[0][0]
        self.assertEqual(tuplify(self.dd[[2, 0], 'name']), ('Bob', 'Alice'))
        self.assertEqual(tuplify(self.dd[[2, 0], 0]), ('Bob', 'Alice'))
        self.assertEqual(tuplify(self.dd[[2, 0], [1, 0]]), ((10, 'Bob'),
                                                   (100, 'Alice')))

    def test_laziness(self):
        assert isinstance(self.dd[:, 'name'], Iterator)

class Test_StreamingTransfer(AccountTestData):
    def test_init(self):
        with filetext(self.text) as fn:
            dd = JSON_Streaming(fn, schema=self.schema)
            self.assertEquals(tuple(dd), self.tuples)
            assert dd.dshape in set((
                datashape.dshape('var * {name: string, amount: int32}'),
                datashape.dshape('5 * {name: string, amount: int32}')))


    def test_chunks(self):
        with filetext(self.text) as fn:
            dd = JSON_Streaming(fn, schema=self.schema)
            chunks = list(dd.chunks(blen=2))
            assert isinstance(chunks[0], nd.array)
            self.assertEquals(len(chunks), 3)
            self.assertEquals(nd.as_py(chunks[0]), self.dicts[:2])


    def test_append(self):
        with filetext('') as fn:
            dd = JSON_Streaming(fn, mode='w', schema=self.schema)
            dd.extend([self.tuples[0]])
            with open(fn) as f:
                self.assertEquals(json.loads(f.read().strip()), self.dicts[0])

            self.assertRaises(ValueError, lambda : dd.extend([5.5]))
            self.assertRaises(ValueError,
                              lambda : dd.extend([{'name': 5, 'amount': 1.3}]))

    def test_extend_dicts(self):
        with filetext('') as fn:
            dd = JSON_Streaming(fn, mode='r+', schema=self.schema)
            dd.extend(self.dicts)
            self.assertEquals(tuplify(tuple(dd)), self.tuples)

    def test_extend_tuples(self):
        with filetext('') as fn:
            dd = JSON_Streaming(fn, mode='r+', schema=self.schema)
            dd.extend(self.tuples)
            self.assertEquals(tuplify(tuple(dd)), self.tuples)

    def test_getitem(self):
        with filetext(self.text) as fn:
            dd = JSON_Streaming(fn, mode='r', schema=self.schema)
            self.assertEqual(tuplify(dd[0]), self.tuples[0])
            self.assertEqual(tuplify(dd[2:4]), self.tuples[2:4])

    def test_as_dynd(self):
        with filetext(self.text) as fn:
            dd = JSON_Streaming(fn, mode='r', schema=self.schema)
            assert nd.as_py(dd.as_dynd()) == self.dicts

    def test_as_py(self):
        with filetext(self.text) as fn:
            dd = JSON_Streaming(fn, mode='r', schema=self.schema)
            self.assertEqual(dd.as_py(), self.tuples)

if __name__ == '__main__':
    unittest.main()
