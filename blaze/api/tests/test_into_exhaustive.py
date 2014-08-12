from __future__ import absolute_import, division, print_function

from dynd import nd
import numpy as np
from pandas import DataFrame

from blaze.api.into import into, discover
from datashape import dshape
from blaze.bcolz import *
import blaze
from blaze import Table
import bcolz


L = [[100, 1, 'Alice'],
     [200, 2, 'Bob'],
     [300, 3, 'Charlie']]

df = DataFrame(L, columns=['amount', 'id', 'name'])

x = np.array(list(map(tuple, L)), dtype=[('amount', 'i8'), ('id', 'i8'), ('name', 'U7')])

arr = nd.array(L, dtype='{amount: int64, id: int64, name: string}')

bc = bcolz.ctable([np.array([100, 200, 300], dtype=np.int64),
                   np.array([1, 2, 3], dtype=np.int64),
                   np.array(['Alice', 'Bob', 'Charlie'], dtype='U7')],
                  names=['amount', 'id', 'name'])


def normalize(a):
    return str(a).replace("u'", "'").replace("(", "[").replace(")", "]")

def test_base():
    A = [Table(L, schema='{amount: int64, id: int64, name: string[7]}'),
         df, x, arr, bc]
    B = [L, df, x, arr, bc]
    for a in A:
        for b in B:
            assert normalize(into(type(b), a)) == normalize(b)
