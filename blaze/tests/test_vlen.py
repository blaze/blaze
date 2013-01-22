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

    for i, v in enumerate(c):
        assert v[0] == i
        assert v[1] == str(i*.2)

def test_object_unicode():
    ds = blaze.dshape('x, blob')
    c = blaze.Array([u'a'*i for i in range(10)], ds)

    for i, v in enumerate(c):
        # The outcome are 0-dim arrays (that might change in the future)
        assert v[()] == u'a'*i

def test_object_persistent_blob():
    td = tempfile.mkdtemp()
    tmppath = os.path.join(td, 'c')

    ds = blaze.dshape('x, blob')
    c = blaze.Array([(i, str(i*.2)) for i in range(10)], ds,
                    params=blaze.params(storage=tmppath))

    for i, v in enumerate(c):
        assert v[0] == i
        assert v[1] == str(i*.2)

    # Remove everything under the temporary dir
    shutil.rmtree(td)

def test_object_persistent_blob_reopen():
    td = tempfile.mkdtemp()
    tmppath = os.path.join(td, 'c')

    ds = blaze.dshape('x, blob')
    c = blaze.Array([(i, "s"*i) for i in range(10)], ds,
                    params=blaze.params(storage=tmppath))

    c2 = blaze.open(tmppath)

    for i, v in enumerate(c2):
        assert v[0] == i
        assert v[1] == "s"*i

    # Remove everything under the temporary dir
    shutil.rmtree(td)

def test_intfloat_blob():
    ds = blaze.dshape('x, blob')
    c = blaze.Array([(i, i*.2) for i in range(10)], ds)

    for i, v in enumerate(c):
        print "v:", v, v[0], type(v[0])
        assert v[0] == i
        assert v[1] == i*.2
