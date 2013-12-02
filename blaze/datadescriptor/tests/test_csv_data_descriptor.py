import unittest
import sys
import io
import blaze
from blaze import datashape
from blaze.datadescriptor import (
    CSVDataDescriptor, DyNDDataDescriptor, IDataDescriptor, dd_as_py)
from blaze.py2help import _inttypes, izip
import ctypes

from dynd import nd, ndt

# A CSV toy example
csv_buf = u"""k1,v1,1,False
k2,v2,2,True
k3,v3,3,False
"""
csv_file = io.StringIO(csv_buf)
csv_schema = "{ f0: string; f1: string; f2: int16; f3: bool }"

class TestCSVDataDescriptor(unittest.TestCase):
    def test_basic_object_type(self):
        self.assertTrue(issubclass(CSVDataDescriptor, IDataDescriptor))
        dd = CSVDataDescriptor(csv_file, csv_schema)
        self.assertTrue(isinstance(dd, IDataDescriptor))
        self.assertEqual(dd_as_py(dd), [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False},
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True},
            {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False}])

    def test_iter(self):
        dd = CSVDataDescriptor(csv_file, csv_schema)
        # This equality does not work yet
        # self.assertEqual(dd.dshape, datashape.dshape(
        #     'Var, %s' % csv_schema))

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
        dd = CSVDataDescriptor(csv_file, csv_schema)

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
        dd = CSVDataDescriptor(csv_file, csv_schema)
        vals = []
        for el in dd.iterchunks(blen=2, start=1):
            vals.extend(dd_as_py(el))
        self.assertEqual(vals, [
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True},
            {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False}])

    def test_iterchunks_stop(self):
        dd = CSVDataDescriptor(csv_file, csv_schema)
        vals = []
        for el in dd.iterchunks(blen=1, stop=2):
            vals.append(dd_as_py(el))
        self.assertEqual(vals, [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False},
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True}])

    def test_iterchunks_start_stop(self):
        dd = CSVDataDescriptor(csv_file, csv_schema)
        vals = []
        for el in dd.iterchunks(blen=1, start=1, stop=2):
            vals.append(dd_as_py(el))
        self.assertEqual(vals, [
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True}])


if __name__ == '__main__':
    unittest.main()
