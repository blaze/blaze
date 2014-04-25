import gzip
import json
from functools import partial
from unittest import TestCase
from datashape import Var

from blaze.data import *
from blaze.utils import filetexts

data = {'a.json': {u'name': u'Alice', u'amount': 100},
        'b.json': {u'name': u'Bob', u'amount': 200},
        'c.json': {u'name': u'Charlie', u'amount': 50}}

texts = dict((fn, json.dumps(val)) for fn, val in data.items())

dshape = '{name: string, amount: int}'

class Test_Integrative(TestCase):
    def test_gzip_json_files(self):
        with filetexts(texts, open=gzip.open) as filenames:
            dd = Files(sorted(filenames),
                       JSON,
                       open=gzip.open,
                       subdshape=dshape)

            self.assertEqual(sorted(dd), sorted(data.values()))

            self.assertEqual(dd.dshape, Var() * dshape)
