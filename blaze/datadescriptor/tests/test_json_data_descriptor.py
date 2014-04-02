from __future__ import absolute_import, division, print_function

import unittest
import os
import tempfile

from blaze.datadescriptor import (
    JSON_DDesc, DyND_DDesc, DDesc, ddesc_as_py)

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
        vals = []
        for el in dd:
            self.assertTrue(isinstance(el, DyND_DDesc))
            self.assertTrue(isinstance(el, DDesc))
            vals.append(ddesc_as_py(el))
        self.assertEqual(vals, [1, 2, 3, 4, 5])

    def test_getitem(self):
        dd = JSON_DDesc(self.json_file, schema=json_schema)
        el = dd[1:3]
        self.assertTrue(isinstance(el, DyND_DDesc))
        vals = ddesc_as_py(el)
        self.assertEqual(vals, [2,3])


if __name__ == '__main__':
    unittest.main()
