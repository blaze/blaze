import numpy as np
from ndtable import NDTable
from datashape import datashape
from blaze import Blaze

def setUp():
    na = np.ones((5,5), dtype=np.dtype('int32'))
    na.tofile('foo.npy')

def test_ndable_init():
    blaze = Blaze()

    bi = blaze.open('file://foo.npy')
    table = NDTable(bi, datashape('5, 5, int32'))
