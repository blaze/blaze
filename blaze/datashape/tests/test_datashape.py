from blaze.datashape import *

a = Integer(800)
b = Integer(600)
c = Integer(300)

def test_associative():
    sig1 = a * (b * int64)
    sig2 = (a * b) * int64
    sig3 = a * b * int64
    assert sig1.parameters == sig2.parameters == sig3.parameters

def test_associative2():
    sig1 = a * (b * c * (int64))
    sig2 = ((a * b) * c) * int64
    sig3 = a * b * c * int64
    sig4 = a * ( b * c ) * int64
    assert sig1.parameters == sig2.parameters == sig3.parameters == sig4.parameters

def test_coersion():
    sig1 = int64*2
    sig2 = int64*3*2

    assert type(sig1[0]) is Integer
    assert type(sig2[0]) is Integer

def test_fromlist():
    it = (a, b, int64)
    ds = DataShape(parameters=it)

    x,y,z = tuple(ds)
    assert all([x is a, y is b, z is int64])

def test_fromlist_compose():
    it1 = (a, b)
    it2 = (int64, )
    ds1 = DataShape(parameters=it1)
    ds2 = DataShape(parameters=it2)

    ds = ds2 * ds1

    assert ds[2] is int64

def test_fromlist_compose2():
    it1 = (a, b)
    it2 = (int64, )
    ds1 = DataShape(parameters=it1)
    ds2 = DataShape(parameters=it2)

    ds_x = ds2 * ds1
    ds_y = DataShape(parameters=(ds1, ds2))
    assert list(ds_x) == list(ds_y)

def test_iteration():
    ds = DataShape(parameters=[a,a,a])

    ds2 = int64 * (ds * ds)
    assert ds2[0] == a
    assert ds2[1] == a
    assert ds2[-1] is int64
