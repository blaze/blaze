from __future__ import absolute_import, division, print_function

import unittest
import os
import tempfile
import json
from dynd import nd
import datashape
from blaze.datadescriptor.as_py import ddesc_as_py

from blaze.data import JSON, JSON_Streaming
from blaze.utils import filetext, raises

# TODO: This isn't actually being used!

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
            f.write(json.dumps(self.data))

    def tearDown(self):
        os.remove(self.filename)

    def test_basic(self):
        dd = JSON(self.filename, 'r', dshape=self.dshape)
        self.assertEqual(list(dd),
                         [nd.as_py(nd.parse_json(self.dshape,
                             json.dumps(self.data)))])


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
        print(list(dd))
        self.assertEqual(list(dd), [1, 2, 3, 4, 5])

class Test_StreamingTransfer(unittest.TestCase):

    data = [{'name': 'Alice', 'amount': 100},
            {'name': 'Alice', 'amount': 50},
            {'name': 'Bob', 'amount': 10},
            {'name': 'Charlie', 'amount': 200},
            {'name': 'Bob', 'amount': 100}]

    text = '\n'.join(map(json.dumps, data))

    schema = '{name: string, amount: int32}'

    def test_init(self):
        with filetext(self.text) as fn:
            dd = JSON_Streaming(fn, schema=self.schema)
            self.assertEquals(list(dd), self.data)
            assert dd.dshape in set((
                datashape.dshape('var * {name: string, amount: int32}'),
                datashape.dshape('5 * {name: string, amount: int32}')))


    def test_chunks(self):
        with filetext(self.text) as fn:
            dd = JSON_Streaming(fn, schema=self.schema)
            chunks = list(dd.chunks(blen=2))
            assert isinstance(chunks[0], nd.array)
            self.assertEquals(len(chunks), 3)
            self.assertEquals(nd.as_py(chunks[0]), self.data[:2])


    def test_append(self):
        with filetext('') as fn:
            dd = JSON_Streaming(fn, mode='w', schema=self.schema)
            dd.extend([self.data[0]])
            with open(fn) as f:
                self.assertEquals(json.loads(f.read().strip()), self.data[0])

            self.assertRaises(ValueError, lambda : dd.extend([5.5]))
            self.assertRaises(ValueError,
                              lambda : dd.extend([{'name': 5, 'amount': 1.3}]))

    def test_extend(self):
        with filetext('') as fn:
            dd = JSON_Streaming(fn, mode='r+', schema=self.schema)
            dd.extend(self.data)

            self.assertEquals(list(dd), self.data)

    def test_getitem(self):
        with filetext(self.text) as fn:
            dd = JSON_Streaming(fn, mode='r', schema=self.schema)
            assert dd[0] == self.data[0]
            assert dd[2:4] == self.data[2:4]

    def test_dynd_arr(self):
        with filetext(self.text) as fn:
            dd = JSON_Streaming(fn, mode='r', schema=self.schema)
            assert nd.as_py(dd.dynd_arr()) == self.data

if __name__ == '__main__':
    unittest.main()
