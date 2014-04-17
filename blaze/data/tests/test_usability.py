from unittest import TestCase
import os
from tempfile import mktemp
import gzip

from blaze.utils import filetext
from blaze.data import *

class TestResource(TestCase):
    def setUp(self):
        self.filename = mktemp()

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_resource_csv(self):
        with filetext('1,1\n2,2', extension='.csv') as fn:
            dd = resource(fn, schema='2 * int')
            assert isinstance(dd, CSV)
            self.assertEqual(list(dd), [[1, 1], [2, 2]])

    def test_resource_json(self):
        with filetext('[[1,1], [2,2]]', extension='.json') as fn:
            dd = resource(fn, schema='2 * int')
            assert isinstance(dd, JSON)
            self.assertEqual(list(dd), [[1, 1], [2, 2]])

    def test_resource_gz(self):
        with filetext('1,1\n2,2', extension='.csv.gz', open=gzip.open) as fn:
            dd = resource(fn, schema='2 * int')
            assert isinstance(dd, CSV)
            self.assertEqual(dd.open, gzip.open)
            self.assertEqual(list(dd), [[1, 1], [2, 2]])
