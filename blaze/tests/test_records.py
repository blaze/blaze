from blaze import RecordDecl
from blaze import float32, int32
from numpy import dtype

class Simple(RecordDecl):
    foo = int32
    bar = float32

    __dummy = True

def test_to_numpy():
    converted = Simple.to_dtype()
    assert converted == dtype([('foo', '<i4'), ('bar', '<f4')])
