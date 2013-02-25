# adapted from samples/dot_example.py

import blaze
from blaze.algo.linalg import dot
from blaze.test_utils import assert_raises

def test_dot():
    '''Test of 2D dot product'''
    a = blaze.ones(blaze.dshape('20, 20, float64'))
    b = blaze.ones(blaze.dshape('20, 30, float64'))
    # Do not write output array to disk
    out = dot(a, b, outname=None)

    expected_ds = blaze.dshape('20, 30, float64')
    assert out.datashape._equal(expected_ds)
    # FIXME: Slow, but no other way to do this with Array API implemented so far
    for row in out:
        for elem in row:
            assert abs(elem - 20.0) < 1e-8

def test_dot_not2d_exception():
    '''Dot product of arrays other than 2D should raise exception.'''
    a = blaze.ones(blaze.dshape('20, 20, 20, float64'))
    b = blaze.ones(blaze.dshape('20, 20, 20, float64'))

    with assert_raises(ValueError):
        out = dot(a, b, outname=None)

def test_dot_shape_exception():
    '''Dot product with wrong inner dimensions should raise exception.'''
    a = blaze.ones(blaze.dshape('20, 20, float64'))
    b = blaze.ones(blaze.dshape('30, 30, float64'))

    with assert_raises(ValueError):
        out = dot(a, b, outname=None)

def test_dot_out_exception():
    '''Output array of wrong size should raise exception.'''
    a = blaze.ones(blaze.dshape('20, 20, float64'))
    b = blaze.ones(blaze.dshape('20, 30, float64'))
    out = blaze.zeros(blaze.dshape('20, 20, float64'))

    with assert_raises(ValueError):
        dot(a, b, out=out)
