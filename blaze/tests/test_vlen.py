"""
Tests for the blob data type.
"""

import blaze

def test_simple_blob():
    ds = blaze.dshape('x, blob')

    c = blaze.Array(["s1", "sss2"], ds)

    assert c[0] == "s1"
    assert c[1] == "sss2"

def test_object_blob():
    ds = blaze.dshape('x, blob')

    c = blaze.Array([(i, str(i*.2)) for i in range(10)], ds)
    print "c:", `c`

    for v in c:
        assert v[0] == i
        assert v[1] == i*.2

def test_intfloat_blob():
    ds = blaze.dshape('x, blob')

    c = blaze.Array([(i, i*.2) for i in range(10)], ds)

    for v in c:
        assert v[0] == i
        assert v[1] == i*.2
