from blaze.expr import *


def test_reduction_dshape():
    x = Symbol('x', '5 * 3 * float32')
    assert x.sum().dshape == x.schema
    assert x.sum(axis=0).dshape == dshape('3 * float32')
    assert x.sum(axis=1).dshape == dshape('5 * float32')
    assert x.sum(axis=(0, 1)).dshape == dshape('float32')


def test_keepdims():
    x = Symbol('x', '5 * 3 * float32')
    assert x.sum(axis=0, keepdims=True).dshape == dshape('1 * 3 * float32')
    assert x.sum(axis=1, keepdims=True).dshape == dshape('5 * 1 * float32')
    assert x.sum(axis=(0, 1), keepdims=True).dshape == dshape('1 * 1 * float32')
