import numpy as np
from ndtable.datashape import datashape
from ndtable.table import NDTable
from ndtable.sources.hdf5 import HDF5Source

from tables.description import *

class Table1(IsDescription):
    name     = StringCol(itemsize = 16)
    pressure = Float32Col(shape = (2,3))
    temp     = Float64Col(shape = (2,3))

class Table2(IsDescription):
    name = StringCol(itemsize = 16)
    x    = Float32Col()
    y    = Float32Col()

table1_ds = """N, Record(
    name     = string,
    pressure = float32,
    temp     = float64
)
"""

table1_denormal = """N, Record(
    name     = string,
    pressure = float32,
    temp     = float64,
    x        = float32,
    y        = float32
)
"""

table1_normal = """N, Record(
    name     = string,
    pressure = float32,
    temp     = float64,
    pos      = Record(
        name     = string,
        pressure = float32,
        temp     = float64
    )
)
"""

def test_hdf5_create_regular():
    # Concatentation of two regular tables into one Blaze DataTable

    ai = HDF5Source(Table1, path='example.h5/' 'a')
    bi = HDF5Source(Table1, path='example.h5/' 'b')

    shape = datashape(table1_ds)
    table = NDTable.from_providers(shape, ai, bi)

    assert table.space.regular
    assert table.space.covers

def test_hdf5_create_irregular():
    # Concatentation of two irregular tables into one Blaze DataTable
    # using a join on the ``name`` column inferred from the outer
    # datashape.

    ci = HDF5Source(Table1, path='example.h5/' 'c')
    di = HDF5Source(Table2, path='example.h5/' 'd')

    shape = datashape(table1_ds)
    table = NDTable.from_providers(shape, ci, di)

    assert not table.space.regular

#def test_hdf5_create_snowflake():

    #ci = HDF5Source(Table1, path='example.h5/' 'c')
    #di = HDF5Source(Table2, path='example.h5/' 'd')

    #shape = datashape(table1_ds)
    #table = NDTable.from_providers(shape, ci, di)

def test_hdf5_indexing():
    # Concatentation of two regular tables into one Blaze DataTable

    ai = HDF5Source(Table1, path='example.h5/' 'a')
    bi = HDF5Source(Table1, path='example.h5/' 'b')

    shape = datashape(table1_ds)
    table = NDTable.from_providers(shape, ai, bi)

    #result = table[0]
    #import pdb; pdb.set_trace()
