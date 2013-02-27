"""A sample involving creation of persistent tables
"""

import os
from blaze import Table, fromiter, mean, std, params, open
from random import random

def build_table(table_name, rows):
    """build the table to use in our example.

    if already built just open it"""
    if not os.path.exists(table_name):
        ds = '(x, {i: int64, f: float64})'
        p = params(clevel=5, storage=table_name)
        t = Table([], dshape=ds, params=p)
        for i in xrange(rows):
            t.append((i, random()))

        t.commit()
    else:
        t = open(table_name)

    return t


def build_array(array_name, rows):
    if not os.path.exists(array_name):
        ds = '(x, float)'

        p = params(clevel=5, storage=array_name)
        t = fromiter((0.1*i for i in xrange(rows)),
                     dshape=ds, params=p)
        t.commit()
    else:
        t = open(array_name)
    
    return t

def test_simple():
    table_name = './sample_tables/test_table'
#    array_name = './sample_tables/test_array'

    t = build_table(table_name, 100000)
#    a = build_array(array_name, 100000)

    print t
#    print a.datashape

if __name__ == '__main__':
    test_simple()


## Local Variables:
## mode: python
## coding: utf-8 
## py-indent-offset: 4
## tab-with: 4
## fill-column: 66
## End:
