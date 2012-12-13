"""
Tests for the simple sessions we put on the documentation page.
"""

def test_simple_session():
    from blaze import Array, dshape
    ds = dshape('2, 2, int')

    a = Array([1,2,3,4], ds)

def test_simple_persistence():
    import tempfile, shutil, os.path
    import numpy as np
    from blaze import Array, dshape, params
    ds = dshape('2, 2, float64')
    data = np.zeros(4).reshape(2,2)
    td = tempfile.mkdtemp()
    tmppath = os.path.join(td, 'a')

    a = Array([1,2,3,4], ds, params=params(storage=tmppath))

    # Remove everything under the temporary dir
    shutil.rmtree(td)

def test_custom_dshape():
    from blaze import Array, RecordDecl
    from blaze import int32, string

    class CustomStock(RecordDecl):
        name   = string
        max    = int32
        min    = int32

        def mid(self):
            return (self.min + self.max)/2

    a = Array([('GOOG', 120, 153)], CustomStock)

def test_sqlite():
    from blaze import open
    a = open('sqlite://')
