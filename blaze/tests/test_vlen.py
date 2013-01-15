"""
Tests for the blob data type.
"""

import blaze
import tempfile, shutil, os.path


def test_simple_blob():
    ds = blaze.dshape('x, blob')
    c = blaze.Array(["s1", "sss2"], ds)

    assert c[0] == "s1"
    assert c[1] == "sss2"

def test_simple_persistent_blob():
    td = tempfile.mkdtemp()
    tmppath = os.path.join(td, 'c')

    ds = blaze.dshape('x, blob')
    c = blaze.Array(["s1", "sss2"], ds,
                    params=blaze.params(storage=tmppath))

    assert c[0] == "s1"
    assert c[1] == "sss2"

    # Remove everything under the temporary dir
    shutil.rmtree(td)

def test_object_blob():
    ds = blaze.dshape('x, blob')
    c = blaze.Array([(i, str(i*.2)) for i in range(10)], ds)

    for v in c:
        assert v[0] == i
        assert v[1] == i*.2

def test_object_persistent_blob():
    td = tempfile.mkdtemp()
    tmppath = os.path.join(td, 'c')

    ds = blaze.dshape('x, blob')
    c = blaze.Array([(i, str(i*.2)) for i in range(10)], ds,
                    params=blaze.params(storage=tmppath))

    for v in c:
        assert v[0] == i
        assert v[1] == i*.2

    # Remove everything under the temporary dir
    shutil.rmtree(td)

def test_intfloat_blob():
    ds = blaze.dshape('x, blob')
    c = blaze.Array([(i, i*.2) for i in range(10)], ds)

    for v in c:
        assert v[0] == i
        assert v[1] == i*.2
