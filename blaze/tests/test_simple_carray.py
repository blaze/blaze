# Simple use case for connecting CArray API and Blaze
# Byte Providers.

from blaze import Array, dshape, params, open
import numpy as np

#------------------------------------------------------------------------
# Case
#------------------------------------------------------------------------

# We use the Array object which is immediete in all operations.

def arr():
    return Array([1,2,3], '3, int32')

def ndarr():
    return np.arange(1e5).reshape(1e3, 1e2)

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

# For now, just assert that these operations are implemented.
# We'll implement equality check letter and add assertions that
# check the end result.

# Case 0
def test_getitem():
    nd = arr()
    assert nd[0] == 1
    assert nd[1] == 2
    assert nd[2] == 3

## Case 1
def test_setitem():
    nd = arr()

    for i in xrange(3):
        # Simple write, forces the manifest bytes
        nd[i] = i

    assert(list(nd.data.ca) == range(3))

# Case 2
def test_getslice():
    nd = arr()

    for i in xrange(3):
        # Simple read slice
        len(nd[0:i]) == i

## Case 3
def test_setslice():
    nd = arr()

    for i in xrange(3):
        # Simple write slice
        nd[0:i] = i
        assert list(nd.data.ca)[0:i] == [i]*i

## Case 4
def test_fancyslice():
    nd = arr()

    for i in xrange(3):
        # Simple read fancy slice
        nd[i::2]

# Case nd
def test_getitem_nd():
    # create
    nd = ndarr()
    barray = Array(nd)

    # read
    data = barray[:]

    assert np.all(data == nd)

# Case nd (persistent version)
def test_getitem_nd_persistent():
    import tempfile, shutil, os.path

    td = tempfile.mkdtemp()
    path = os.path.join(td, 'test.blz')

    # write
    bparams = params(storage=path, clevel=6)
    nd = ndarr()
    barray = Array(nd, params=bparams)

    # read
    arr = open(path)
    data = arr[:]

    assert np.all(data == nd)

    shutil.rmtree(td)
