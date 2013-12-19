import unittest
import sys
import io
import os
import tempfile
import blaze
from blaze import datashape
from blaze.datadescriptor import (
    JSONDataDescriptor, DyNDDataDescriptor, IDataDescriptor, dd_as_py)
from blaze.py2help import _inttypes, izip
import ctypes

from dynd import nd, ndt

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

_json_schema = """{
  type: string;
  images: var,
   Image: {
        Width: int16;
        Height: int16;
        Title: string;
        Thumbnail: {
            Url: string;
            Height: int16;
            Width: int16;
        }
        IDs: var, int32;
    };
}
"""

json_buf = u"[1, 2, 3, 4, 5]"
json_schema = "var, int8"

class TestJSONDataDescriptor(unittest.TestCase):

    def setUp(self):
        self.json_file = tempfile.mktemp(".json")
        with file(self.json_file, "w") as f:
            f.write(json_buf)

    def tearDown(self):
        os.remove(self.json_file)

    def test_basic_object_type(self):
        self.assertTrue(issubclass(JSONDataDescriptor, IDataDescriptor))
        dd = JSONDataDescriptor(self.json_file, schema=json_schema)
        self.assertTrue(isinstance(dd, IDataDescriptor))
        self.assertEqual(dd_as_py(dd), [1, 2, 3, 4, 5])

    def test_iter(self):
        dd = JSONDataDescriptor(self.json_file, schema=json_schema)
        # This equality does not work yet
        # self.assertEqual(dd.dshape, datashape.dshape(
        #     'Var, %s' % json_schema))

        # Iteration should produce DyNDDataDescriptor instances
        vals = []
        for el in dd:
            self.assertTrue(isinstance(el, DyNDDataDescriptor))
            self.assertTrue(isinstance(el, IDataDescriptor))
            vals.append(dd_as_py(el))
        self.assertEqual(vals, [1, 2, 3, 4, 5])

    def test_getitem(self):
        dd = JSONDataDescriptor(self.json_file, schema=json_schema)
        el = dd[1:3]
        self.assertTrue(isinstance(el, DyNDDataDescriptor))
        vals = dd_as_py(el)
        self.assertEqual(vals, [2,3])


if __name__ == '__main__':
    unittest.main()
