from __future__ import absolute_import, division, print_function

import os
import tempfile
import unittest

import blaze
from blaze.py2help import skip, skipIf
from blaze.datadescriptor import dd_as_py
from blaze.tests.common import MayBePersistentTest
from blaze import (append,
    DyND_DDesc, BLZ_DDesc, HDF5_DDesc,
    CSV_DDesc, JSON_DDesc)

from blaze.optional_packages import tables_is_here
if tables_is_here:
    import tables as tb


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
        ddesc = CSV_DDesc(self.fname, mode='r', schema=csv_schema)
        a = blaze.array(ddesc)
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), csv_ldict)

    def test_from_dialect(self):
        ddesc = CSV_DDesc(self.fname, mode='r',
                          schema=csv_schema, dialect='excel')
        a = blaze.array(ddesc)
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), csv_ldict)

    def test_from_has_header(self):
        ddesc = CSV_DDesc(
            self.fname, mode='r', schema=csv_schema, has_header=False)
        a = blaze.array(ddesc)
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), csv_ldict)

    def test_append(self):
        ddesc = CSV_DDesc(self.fname, mode='r+', schema=csv_schema)
        a = blaze.array(ddesc)
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
        ddesc = JSON_DDesc(self.fname, mode='r', schema=json_schema)
        a = blaze.array(ddesc)
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [1, 2, 3, 4, 5])


class TestOpenBLZ(MayBePersistentTest, unittest.TestCase):

    disk = True
    dir_ = True

    def test_open(self):
        ddesc = BLZ_DDesc(path=self.rootdir, mode='w')
        self.assertTrue(ddesc.mode == 'w')
        a = blaze.ones('0 * float64', ddesc=ddesc)
        append(a,range(10))
        # Re-open the dataset
        ddesc = BLZ_DDesc(path=self.rootdir, mode='r')
        self.assertTrue(ddesc.mode == 'r')
        a2 = blaze.array(ddesc)
        self.assertTrue(isinstance(a2, blaze.Array))
        self.assertEqual(dd_as_py(a2._data), list(range(10)))

    def test_wrong_open_mode(self):
        ddesc = BLZ_DDesc(path=self.rootdir, mode='w')
        a = blaze.ones('10 * float64', ddesc=ddesc)
        # Re-open the dataset
        ddesc = BLZ_DDesc(path=self.rootdir, mode='r')
        self.assertTrue(ddesc.mode == 'r')
        a2 = blaze.array(ddesc)
        self.assertRaises(IOError, append, a2, [1])


class TestOpenHDF5(MayBePersistentTest, unittest.TestCase):

    disk = True

    @skipIf(not tables_is_here, 'pytables is not installed')
    def test_open(self):
        ddesc = HDF5_DDesc(path=self.file, datapath='/earray', mode='a')
        self.assertTrue(ddesc.mode == 'a')
        a = blaze.ones('0 * float64', ddesc=ddesc)
        append(a,range(10))
        # Re-open the dataset in URI
        ddesc = HDF5_DDesc(path=self.file, datapath='/earray', mode='r')
        self.assertTrue(ddesc.mode == 'r')
        a2 = blaze.array(ddesc)
        self.assertTrue(isinstance(a2, blaze.Array))
        self.assertEqual(dd_as_py(a2._data), list(range(10)))

    @skipIf(not tables_is_here, 'pytables is not installed')
    def test_wrong_open_mode(self):
        ddesc = HDF5_DDesc(path=self.file, datapath='/earray', mode='w')
        a = blaze.ones('10 * float64', ddesc=ddesc)
        # Re-open the dataset
        ddesc = HDF5_DDesc(path=self.file, datapath='/earray', mode='r')
        self.assertTrue(ddesc.mode == 'r')
        a2 = blaze.array(ddesc)
        self.assertRaises(tb.FileModeError, append, a2, [1])


if __name__ == '__main__':
   unittest.main(verbosity=2)
