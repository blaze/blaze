from __future__ import print_function, absolute_import

import blaze
from blaze.datadescriptor import dd_as_py
from blaze.eval import append
import numpy as np
import unittest
import tempfile
import os


# A CSV toy example
csv_buf = u"""k1,v1,1,False
k2,v2,2,True
k3,v3,3,False
"""
csv_schema = "{ f0: string; f1: string; f2: int16; f3: bool }"


class TestOpenCSV(unittest.TestCase):

    def setUp(self):
        _, self.fname = tempfile.mkstemp(suffix='.csv')
        self.url = "csv:///" + self.fname
        with file(self.fname, "wb") as f:
            f.write(csv_buf)

    def tearDown(self):
        os.unlink(self.fname)

    def test_open(self):
        store = blaze.Storage(self.url, mode='r')
        a = blaze.open(store, csv_schema)
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False},
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True},
            {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False}])

    def test_append(self):
        store = blaze.Storage(self.url, mode='r+')
        a = blaze.open(store, csv_schema)
        append(a, ["k4", "v4", 4, True])
        self.assertEqual(dd_as_py(a._data), [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False},
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True},
            {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False},
            {u'f0': u'k4', u'f1': u'v4', u'f2': 4, u'f3': True}])


if __name__ == '__main__':
    unittest.main(verbosity=2)
