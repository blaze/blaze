from __future__ import absolute_import, division, print_function

import os
import tempfile
import unittest

import blaze
from blaze.datadescriptor import dd_as_py
from blaze.tests.common import MayBePersistentTest
from blaze import (append,
    DyNDDataDescriptor, BLZDataDescriptor, HDF5DataDescriptor,
    CSVDataDescriptor, JSONDataDescriptor)


# A CSV toy example
csv_buf = u"""k1,v1,1,False
k2,v2,2,True
k3,v3,3,False
"""
csv_schema = "{ f0: string, f1: string, f2: int16, f3: bool }"
csv_ldict =  [
    {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False},
    {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True},
    {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False}
    ]


class TestOpenCSV(unittest.TestCase):

    def setUp(self):
        handle, self.fname = tempfile.mkstemp(suffix='.csv')
        with os.fdopen(handle, "w") as f:
            f.write(csv_buf)

    def tearDown(self):
        os.unlink(self.fname)

    def test_open(self):
        dd = CSVDataDescriptor(self.fname, mode='r')
        a = blaze.from_csv(dd, schema=csv_schema)
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), csv_ldict)

    def test_from_dialect(self):
        dd = CSVDataDescriptor(self.fname, mode='r')
        a = blaze.from_csv(dd, schema=csv_schema, dialect='excel')
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), csv_ldict)

    def test_from_has_header(self):
        dd = CSVDataDescriptor(self.fname, mode='r')
        a = blaze.from_csv(dd, schema=csv_schema, has_header=False)
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), csv_ldict)

    def test_append(self):
        dd = CSVDataDescriptor(self.fname, mode='r+')
        a = blaze.from_csv(dd, schema=csv_schema)
        blaze.append(a, ["k4", "v4", 4, True])
        self.assertEqual(dd_as_py(a._data), csv_ldict + \
            [{u'f0': u'k4', u'f1': u'v4', u'f2': 4, u'f3': True}])


json_buf = u"[1, 2, 3, 4, 5]"
json_schema = "var * int8"


class TestOpenJSON(unittest.TestCase):

    def setUp(self):
        handle, self.fname = tempfile.mkstemp(suffix='.json')
        with os.fdopen(handle, "w") as f:
            f.write(json_buf)

    def tearDown(self):
        os.unlink(self.fname)

    def test_open(self):
        dd = JSONDataDescriptor(path=self.fname, mode='r')
        a = blaze.from_json(dd, schema=json_schema)
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [1, 2, 3, 4, 5])


class TestOpenBLZ(MayBePersistentTest, unittest.TestCase):

    disk = True

    def test_open(self):
        dd = BLZDataDescriptor(path=self.rootdir, mode='a')
        self.assertTrue(dd.mode == 'a')
        a = blaze.ones('0 * float64', dd=dd)
        append(a,range(10))
        # Re-open the dataset in URI
        dd = BLZDataDescriptor(path=self.rootdir, mode='r')
        self.assertTrue(dd.mode == 'r')
        a2 = blaze.from_blz(dd=dd)
        self.assertTrue(isinstance(a2, blaze.Array))
        self.assertEqual(dd_as_py(a2._data), list(range(10)))


if __name__ == '__main__':
   unittest.main(verbosity=2)
