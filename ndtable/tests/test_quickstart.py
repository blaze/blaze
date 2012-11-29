"""
Tests for the simple sessions we put on the documentation page.
"""

def test_simple_session():
    from ndtable import Array, dshape
    ds = dshape('2, 2, int')

    a = Array([1,2,3,4], ds)

def test_custom_dshape():
    from ndtable import Array, RecordDecl
    from ndtable import int32, string

    class CustomStock(RecordDecl):
        name   = string
        max    = int32
        min    = int32

        def mid(self):
            return (self.min + self.max)/2

    a = Array([('GOOG', 120, 153)], CustomStock)
