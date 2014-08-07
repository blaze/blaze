from __future__ import absolute_import, division, print_function

import pytest
pymongo = pytest.importorskip('pymongo')

from contextlib import contextmanager
from blaze.mongo import *
from toolz.curried import get

conn = pymongo.MongoClient()
db = conn.test_db

@contextmanager
def collection(data=[]):
    coll = db.tmp_collection
    if data:
        coll = into(coll, data)

    try:
        yield coll
    finally:
        coll.drop()


bank = [{'name': 'Alice', 'amount': 100},
        {'name': 'Alice', 'amount': 200},
        {'name': 'Bob', 'amount': 100},
        {'name': 'Bob', 'amount': 200},
        {'name': 'Bob', 'amount': 300}]

def test_discover():
    with collection(bank) as coll:
        assert discover(coll) == dshape('5 * {amount: int64, name: string}')


def test_into():
    with collection([]) as coll:
        key = get(['name', 'amount'])
        assert sorted(into([], into(coll, bank)), key=key) == \
                sorted(bank, key=key)
