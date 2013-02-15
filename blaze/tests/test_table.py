from blaze import dshape
from blaze import NDTable, Table, NDArray, Array

def test_arrays():
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


def test_record():
    expected_ds = dshape('1, {x: int32; y: float32}')

    t = NDTable([(1, 2.1), (2, 3.1)], dshape='1, {x: int32; y: float32}')
    t.datashape._equal(expected_ds)

    str(t)
    repr(t)

def test_record_consume():
    expected_ds = dshape("4, {i: int64; f: float64}")

    d = {
        'i'   : [1, 2, 3, 4],
        'f'   : [4., 3., 2., 1.]
    }
    t = NDTable(d)
    t.datashape._equal(expected_ds)

def test_record_consume2():
    d = {
        'a'   : ["foo", "bar"],
        'b'   : [4., 3., 2., 1.]
    }
    table = NDTable(d)
