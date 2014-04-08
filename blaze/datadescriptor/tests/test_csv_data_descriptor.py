from __future__ import absolute_import, division, print_function

import unittest
import tempfile
import os
import csv

import datashape

from blaze.datadescriptor import (
    CSV_DDesc, DyND_DDesc, DDesc, ddesc_as_py)


def sanitize(lines):
    return '\n'.join(line.strip() for line in lines.split('\n'))


class TestCSV_DDesc_dialect(unittest.TestCase):

    buf = sanitize(
    u"""Name Amount
        Alice 100
        Bob 200
        Alice 50
    """)

    schema = "{ f0: string, f1: int }"

    def setUp(self):
        handle, self.csv_file = tempfile.mkstemp(".csv")
        with os.fdopen(handle, "w") as f:
            f.write(self.buf)

    def tearDown(self):
        os.remove(self.csv_file)

    def test_overwrite_delimiter(self):
        dd = CSV_DDesc(self.csv_file, dialect='excel', schema=self.schema,
                delimiter=' ')
        assert dd.dialect['delimiter'] == ' '

    def test_content(self):
        dd = CSV_DDesc(self.csv_file, dialect='excel', schema=self.schema,
                delimiter=' ')
        print(ddesc_as_py(dd))
        s = str(ddesc_as_py(dd))
        assert 'Alice' in s and 'Bob' in s


class TestCSV_DDesc(unittest.TestCase):

    # A CSV toy example
    buf = sanitize(
    u"""k1,v1,1,False
        k2,v2,2,True
        k3,v3,3,False
    """)
    schema = "{ f0: string, f1: string, f2: int16, f3: bool }"

    def setUp(self):
        handle, self.csv_file = tempfile.mkstemp(".csv")
        with os.fdopen(handle, "w") as f:
            f.write(self.buf)

    def tearDown(self):
        os.remove(self.csv_file)

    def test_basic_object_type(self):
        self.assertTrue(issubclass(CSV_DDesc, DDesc))
        dd = CSV_DDesc(self.csv_file, schema=self.schema)
        self.assertTrue(isinstance(dd, DDesc))
        self.assertTrue(isinstance(dd.dshape.shape[0], datashape.Var))
        self.assertEqual(ddesc_as_py(dd), [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False},
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True},
            {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False}])

    def test_iter(self):
        dd = CSV_DDesc(self.csv_file, schema=self.schema)

        # Iteration should produce DyND_DDesc instances
        vals = []
        for el in dd:
            self.assertTrue(isinstance(el, DyND_DDesc))
            self.assertTrue(isinstance(el, DDesc))
            vals.append(ddesc_as_py(el))
        self.assertEqual(vals, [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False},
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True},
            {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False}])

    def test_iterchunks(self):
        dd = CSV_DDesc(self.csv_file, schema=self.schema)

        # Iteration should produce DyND_DDesc instances
        vals = []
        for el in dd.iterchunks(blen=2):
            self.assertTrue(isinstance(el, DyND_DDesc))
            self.assertTrue(isinstance(el, DDesc))
            vals.extend(ddesc_as_py(el))
        self.assertEqual(vals, [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False},
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True},
            {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False}])

    def test_iterchunks_start(self):
        dd = CSV_DDesc(self.csv_file, schema=self.schema)
        vals = []
        for el in dd.iterchunks(blen=2, start=1):
            vals.extend(ddesc_as_py(el))
        self.assertEqual(vals, [
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True},
            {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False}])

    def test_iterchunks_stop(self):
        dd = CSV_DDesc(self.csv_file, schema=self.schema)
        vals = [ddesc_as_py(v) for v in dd.iterchunks(blen=1, stop=2)]
        self.assertEqual(vals, [
            [{u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False}],
            [{u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True}]])

    def test_iterchunks_start_stop(self):
        dd = CSV_DDesc(self.csv_file, schema=self.schema)
        vals = [ddesc_as_py(v) for v in dd.iterchunks(blen=1, start=1, stop=2)]
        self.assertEqual(vals, [[
            {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True}]])

    def test_append(self):
        # Get a private file so as to not mess the original one
        handle, csv_file = tempfile.mkstemp(".csv")
        with os.fdopen(handle, "w") as f:
            f.write(self.buf)
        dd = CSV_DDesc(csv_file, schema=self.schema, mode='r+')
        dd.append(["k4", "v4", 4, True])
        vals = [ddesc_as_py(v) for v in dd.iterchunks(blen=1, start=3)]
        self.assertEqual(vals, [[
            {u'f0': u'k4', u'f1': u'v4', u'f2': 4, u'f3': True}]])
        os.remove(csv_file)

    def test_getitem_start(self):
        dd = CSV_DDesc(self.csv_file, schema=self.schema)
        el = dd[0]
        self.assertTrue(isinstance(el, DyND_DDesc))
        vals = ddesc_as_py(el)
        self.assertEqual(vals, [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False}])

    def test_getitem_stop(self):
        dd = CSV_DDesc(self.csv_file, schema=self.schema)
        el = dd[:1]
        self.assertTrue(isinstance(el, DyND_DDesc))
        vals = ddesc_as_py(el)
        self.assertEqual(vals, [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False}])

    def test_getitem_step(self):
        dd = CSV_DDesc(self.csv_file, schema=self.schema)
        el = dd[::2]
        self.assertTrue(isinstance(el, DyND_DDesc))
        vals = ddesc_as_py(el)
        self.assertEqual(vals, [
            {u'f0': u'k1', u'f1': u'v1', u'f2': 1, u'f3': False},
            {u'f0': u'k3', u'f1': u'v3', u'f2': 3, u'f3': False}])

    def test_getitem_start_step(self):
        dd = CSV_DDesc(self.csv_file, schema=self.schema)
        el = dd[1::2]
        self.assertTrue(isinstance(el, DyND_DDesc))
        vals = ddesc_as_py(el)
        self.assertEqual(vals, [
        {u'f0': u'k2', u'f1': u'v2', u'f2': 2, u'f3': True}])


if __name__ == '__main__':
    unittest.main()
