from __future__ import absolute_import, division, print_function

import gzip
from functools import partial
from unittest import TestCase
from datashape import var, dshape

from blaze.data import *
from blaze.utils import filetexts
import json

import json

data = {'a.json': {u'name': u'Alice', u'amount': 100},
        'b.json': {u'name': u'Bob', u'amount': 200},
        'c.json': {u'name': u'Charlie', u'amount': 50}}

tuples = (('Alice', 100),
          ('Bob', 200),
          ('Charlie', 50))

texts = dict((fn, json.dumps(val)) for fn, val in data.items())

schema = '{name: string, amount: int}'

class Test_Integrative(TestCase):
    def test_gzip_json_files(self):
        with filetexts(texts, open=gzip.open) as filenames:
            descriptors = [JSON(fn, dshape=schema, open=gzip.open)
                            for fn in sorted(filenames)]
            dd = Stack(descriptors)

            self.assertEqual(sorted(dd), sorted(tuples))

            self.assertEqual(dd.schema, dshape(schema))
