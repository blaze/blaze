import numpy as np
from ndtable.datashape import datashape
from ndtable.table import DataTable


def setUp():
    na = np.ones((5,5), dtype=np.dtype('int32'))
    na.tofile('foo.npy')

def test_from_views():
    # Heterogenous Python lists of objects
    a = [1,2,3,4]
    b = [5,6,7,8]

    shape = datashape('2, 4, int32')
    table = DataTable.from_views(shape, a, b)

def test_from_views_complex_dims():
    a = [1,2,3,4]
    b = [5,6,7,8]

    shape = datashape('2, Var(10), int32')
    table = DataTable.from_views(shape, a, b)


def test_from_views_complex_dims():
    a = [1,2,3,4]
    b = [5,6,7,8]

    shape = datashape('2, Var(10), int32')
    table = DataTable.from_views(shape, a, b)
