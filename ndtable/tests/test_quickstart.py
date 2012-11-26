"""
Tests for the simple sessions we put on the documentation page.
"""

def test_simple_session():
    from ndtable import Array, dshape
    ds = dshape('2, 2, int')

    a = Array([1,2,3,4], ds)
