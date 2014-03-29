from __future__ import absolute_import, division, print_function

import unittest
import tempfile
import os

import datashape

from blaze.datadescriptor import (
    CSVDataDescriptor, DyNDDataDescriptor, IDataDescriptor, dd_as_py)

# A CSV toy example
csv_buf = u"""k1,v1,1,False
k2,v2,2,True
k3,v3,3,False
"""
csv_schema = "{ f0: string, f1: string, f2: int16, f3: bool }"


class TestCSVDataDescriptor(unittest.TestCase):

    def setUp(self):
        handle, self.csv_file = tempfile.mkstemp(".csv")
        with os.fdopen(handle, "w") as f:
            f.write(csv_buf)

    def tearDown(self):
        os.remove(self.csv_file)

    def test_basic_object_type(self):
        self.assertTrue(issubclass(CSVDataDescriptor, IDataDescriptor))
        dd = CSVDataDescriptor(self.csv_file, schema=csv_schema)
        self.assertTrue(isinstance(dd, IDataDescriptor))
        self.assertTrue(isinstance(dd.dshape.shape[0], datashape.Var))
        self.assertEqual(dd_as_py(dd), [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False},
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True},
            {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False}])

    def test_iter(self):
        dd = CSVDataDescriptor(self.csv_file, schema=csv_schema)

        # Iteration should produce DyNDDataDescriptor instances
        vals = []
        for el in dd:
            self.assertTrue(isinstance(el, DyNDDataDescriptor))
            self.assertTrue(isinstance(el, IDataDescriptor))
            vals.append(dd_as_py(el))
        self.assertEqual(vals, [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False},
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True},
            {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False}])

    def test_iterchunks(self):
        dd = CSVDataDescriptor(self.csv_file, schema=csv_schema)

        # Iteration should produce DyNDDataDescriptor instances
        vals = []
        for el in dd.iterchunks(blen=2):
            self.assertTrue(isinstance(el, DyNDDataDescriptor))
            self.assertTrue(isinstance(el, IDataDescriptor))
            vals.extend(dd_as_py(el))
        self.assertEqual(vals, [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False},
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True},
            {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False}])

    def test_iterchunks_start(self):
        dd = CSVDataDescriptor(self.csv_file, schema=csv_schema)
        vals = []
        for el in dd.iterchunks(blen=2, start=1):
            vals.extend(dd_as_py(el))
        self.assertEqual(vals, [
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True},
            {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False}])

    def test_iterchunks_stop(self):
        dd = CSVDataDescriptor(self.csv_file, schema=csv_schema)
        vals = [dd_as_py(v) for v in dd.iterchunks(blen=1, stop=2)]
        self.assertEqual(vals, [
            [{u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False}],
            [{u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True}]])

    def test_iterchunks_start_stop(self):
        dd = CSVDataDescriptor(self.csv_file, schema=csv_schema)
        vals = [dd_as_py(v) for v in dd.iterchunks(blen=1, start=1, stop=2)]
        self.assertEqual(vals, [[
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True}]])

    def test_append(self):
        # Get a private file so as to not mess the original one
        handle, csv_file = tempfile.mkstemp(".csv")
        with os.fdopen(handle, "w") as f:
            f.write(csv_buf)
        dd = CSVDataDescriptor(csv_file, schema=csv_schema, mode='r+')
        dd.append(["k4", "v4", 4, True])
        vals = [dd_as_py(v) for v in dd.iterchunks(blen=1, start=3)]
        self.assertEqual(vals, [[
            {u'f0': u'k4', u'f1': u'v4', u'f2': 4, u'f3': True}]])
        os.remove(csv_file)

    def test_getitem_start(self):
        dd = CSVDataDescriptor(self.csv_file, schema=csv_schema)
        el = dd[0]
        self.assertTrue(isinstance(el, DyNDDataDescriptor))
        vals = dd_as_py(el)
        self.assertEqual(vals, [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False}])

    def test_getitem_stop(self):
        dd = CSVDataDescriptor(self.csv_file, schema=csv_schema)
        el = dd[:1]
        self.assertTrue(isinstance(el, DyNDDataDescriptor))
        vals = dd_as_py(el)
        self.assertEqual(vals, [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False}])

    def test_getitem_step(self):
        dd = CSVDataDescriptor(self.csv_file, schema=csv_schema)
        el = dd[::2]
        self.assertTrue(isinstance(el, DyNDDataDescriptor))
        vals = dd_as_py(el)
        self.assertEqual(vals, [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False},
            {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False}])

    def test_getitem_start_step(self):
        dd = CSVDataDescriptor(self.csv_file, schema=csv_schema)
        el = dd[1::2]
        self.assertTrue(isinstance(el, DyNDDataDescriptor))
        vals = dd_as_py(el)
        self.assertEqual(vals, [
        {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True}])


if __name__ == '__main__':
    unittest.main()
