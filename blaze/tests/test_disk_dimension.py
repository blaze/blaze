import os.path
import shutil
import numpy as np
import blaze as blz

def test_perserve():
    shape = (3,4)
    arr = np.ones(shape)

    dshape = "%s,%s, float64" % (shape[0], shape[1])
    path = "p.blz"
    if os.path.exists(path):
        shutil.rmtree(path)
    bparams = blz.params(storage=path)
    barray = blz.Array(arr, dshape, params=bparams)
    print "barray:", repr(barray)

    barray2 = blz.open(path)
    print "barray2:", repr(barray2)

    assert(str(barray.datashape) == str(barray2.datashape))
