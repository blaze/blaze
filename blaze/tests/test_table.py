import numpy as np
from blaze import dshape
from blaze import NDTable, Table, NDArray, Array

def test_all_construct():
    # Assert that the pretty pritner works for all of the
    # toplevel structures

    expected_ds = dshape('3, int')

    a = NDArray([1,2,3])
    str(a)
    repr(a)
    a.datashape._equal(expected_ds)

    a = Array([1,2,3])
    str(a)
    repr(a)
    a.datashape._equal(expected_ds)


    a = NDTable([(1, 1)])
    str(a)
    repr(a)
    #a.datashape._equal(expected_ds)

    a = Table([(1, 1)])
    str(a)
    repr(a)
    #a.datashape._equal(expected_ds)

def test_record():
    data = NDTable([(1, 2.1)], '1, Record(x=int32, y=float)')
