from __future__ import absolute_import, division, print_function

import unittest
import os
import tempfile
import json
from dynd import nd

from blaze.datadescriptor import (
    JSON_DDesc, DyND_DDesc, DDesc, ddesc_as_py)
from blaze.datadescriptor.util import filetext, raises

# TODO: This isn't actually being used!
_json_buf = u"""{
    "type": "ImageCollection",
    "images": [
        "Image": {
            "Width":  800,
            "Height": 600,
            "Title":  "View from 15th Floor",
            "Thumbnail": {
                "Url":    "http://www.example.com/image/481989943",
                "Height": 125,
                "Width":  "100"
            },
            "IDs": [116, 943, 234, 38793]
        }
    ]
}
"""

# TODO: This isn't actually being used!
_json_schema = """{
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
    };
}
"""


json_buf = u"[1, 2, 3, 4, 5]"
json_schema = "var * int8"


class TestJSON_DDesc(unittest.TestCase):

    def setUp(self):
        handle, self.json_file = tempfile.mkstemp(".json")
        with os.fdopen(handle, "w") as f:
            f.write(json_buf)

    def tearDown(self):
        os.remove(self.json_file)

    def test_basic_object_type(self):
        self.assertTrue(issubclass(JSON_DDesc, DDesc))
        dd = JSON_DDesc(self.json_file, schema=json_schema)
        self.assertTrue(isinstance(dd, DDesc))
        self.assertEqual(ddesc_as_py(dd), [1, 2, 3, 4, 5])

    def test_iter(self):
        dd = JSON_DDesc(self.json_file, schema=json_schema)
        # This equality does not work yet
        # self.assertEqual(dd.dshape, datashape.dshape(
        #     'Var, %s' % json_schema))

        # Iteration should produce DyND_DDesc instances
        self.assertEqual(list(dd), [[1, 2, 3, 4, 5]])

    def test_getitem(self):
        dd = JSON_DDesc(self.json_file, schema=json_schema)
        el = dd[1:3]
        self.assertTrue(isinstance(el, DyND_DDesc))
        vals = ddesc_as_py(el)
        self.assertEqual(vals, [2,3])


data = [{'name': 'Alice', 'amount': 100},
        {'name': 'Alice', 'amount': 50},
        {'name': 'Bob', 'amount': 10},
        {'name': 'Charlie', 'amount': 200},
        {'name': 'Bob', 'amount': 100}]

text = '\n'.join(map(json.dumps, data))

schema = '{name: string, amount: int32}'

def test_init():
    with filetext(text) as fn:
        dd = JSON_DDesc(fn, schema=schema)
        assert list(dd) == data


def test_iterchunks():
    with filetext(text) as fn:
        dd = JSON_DDesc(fn, schema=schema)
        chunks = list(dd.iterchunks(blen=2))
        assert isinstance(chunks[0], nd.array)
        assert len(chunks) == 3
        assert nd.as_py(chunks[0]) == data[:2]


def test_append():
    with filetext('') as fn:
        dd = JSON_DDesc(fn, mode='w', schema=schema)
        dd.append(data[0])
        with open(fn) as f:
            assert json.loads(f.read().strip()) == data[0]

        assert raises(ValueError, lambda : dd.append(5.5))
        assert raises(ValueError, lambda : dd.append({'name': 5, 'amount': 1.3}))

def test_extend():
    with filetext('') as fn:
        dd = JSON_DDesc(fn, mode='w', schema=schema)
        dd.extend(data)

        assert list(dd) == data


if __name__ == '__main__':
    unittest.main()
