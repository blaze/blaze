"""
Test for compatability between NumPy and a subset of Blaze types.
"""

import blaze
import numpy as np

from blaze import dshape
from blaze.datashape.coretypes import NotNumpyCompatible,\
    to_numpy, from_numpy, extract_dims, extract_measure, \
    promote

from nose.tools import assert_raises

#------------------------------------------------------------------------
# To NumPy
#------------------------------------------------------------------------

def test_dtype_compat():
    to_numpy(blaze.int32) == np.int32
    to_numpy(blaze.int64) == np.int64
    to_numpy(blaze.float_) == np.float_
    to_numpy(blaze.int_) == np.int_

def test_shape_compat():
    to_numpy(dshape('1, int32')) == (1,), np.int32
    to_numpy(dshape('1, 2, int32')) == (1, 2), np.int32
    to_numpy(dshape('1, 2, 3, 4, int32')) == (1, 2, 3, 4), np.int32

def test_deconstruct():
    ds = dshape('1, 2, 3, int32')

    extract_dims(ds) == (1,2,3)
    extract_measure(ds) == blaze.int32

def test_not_compat():
    with assert_raises(NotNumpyCompatible):
        to_numpy(dshape('x, int32'))

    with assert_raises(NotNumpyCompatible):
        to_numpy(dshape('{1}, int32'))

    with assert_raises(NotNumpyCompatible):
        to_numpy(dshape('Range(0, 3), int32'))

def test_promote():
    from blaze.expr.graph import IntNode, FloatNode

    # ----------------------------------
    x,y = IntNode(1), FloatNode(1.)
    res = promote(x,y)

    # TODO: check if this is platform specific
    assert res == blaze.float64
    # ----------------------------------
    x,y = IntNode(1), IntNode(1)
    res = promote(x,y)

    assert res == blaze.int32
    # ----------------------------------


#------------------------------------------------------------------------
# From NumPy
#------------------------------------------------------------------------

def test_from_numpy():
    from_numpy((), np.int32) == blaze.int32
    from_numpy((), np.int_) == blaze.int_

    from_numpy((1,), np.int32) == blaze.dshape('1, int32')
    from_numpy((1,2), np.int32) == blaze.dshape('1, 2, int32')
