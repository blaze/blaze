from __future__ import absolute_import, division, print_function

from dynd import nd
import numpy as np
from pandas import DataFrame

from blaze.api.into import into, discover
from datashape import dshape
import blaze
from blaze import Table
import bcolz


L = [[1, 'Alice', 100],
     [2, 'Bob', 200],
     [3, 'Charlie', 300]]

df = DataFrame(L, columns=['id', 'name', 'amount'])

x = np.array(list(map(tuple, L)), dtype=[('id', 'i8'), ('name', 'U7'), ('amount', 'i8')])

arr = nd.array(L, dtype='{id: int64, name: string, amount: int64}')

bc = bcolz.ctable([np.array([1, 2, 3], dtype=np.int64),
                   np.array(['Alice', 'Bob', 'Charlie'], dtype='U7'),
                   np.array([100, 200, 300], dtype=np.int64)],
                  names=['id', 'name', 'amount'])

def test_base():
    A = [Table(L, schema='{id: int64, name: string[7], amount: int64}'),
         df, x, arr, bc]
    B = [L, df, x, arr, bc]
    for a in A:
        for b in B:
            assert str(into(type(b), a)) == str(b)
