import numpy as np
from ndtable.table import NDTable, Index
from ndtable.datashape import datashape
from ndtable.blaze import Blaze

def setUp():
    na = np.ones((5,5), dtype=np.dtype('int32'))
    na.tofile('foo.npy')

def test_ndable_init():
    blaze = Blaze()

    bi = blaze.open('file://foo.npy')
    table = NDTable(bi, datashape('5, 5, int32'))

def test_ndable_init_index():
    blaze = Blaze()

    bi = blaze.open('file://foo.npy')
    idx = Index(bi)
    table = NDTable(idx, datashape('5, 5, int32'))
