from blaze import dshape
from blaze import NDTable, Table, NDArray, Array

def test_all_construct():
    expected_ds = dshape('3, int')

    a = NDTable([1,2,3])
    str(a)
    repr(a)
    a.datashape._equal(expected_ds)

    a = NDArray([1,2,3])
    str(a)
    repr(a)
    a.datashape._equal(expected_ds)

    a = Array([1,2,3])
    str(a)
    repr(a)
    a.datashape._equal(expected_ds)

    #a = Table([1,2,3])
    #str(a)
    #repr(a)
    #a.datashape._equal(expected_ds)
