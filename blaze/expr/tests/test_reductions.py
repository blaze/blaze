from blaze.expr import Symbol, summary
from datashape import dshape


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

    assert x.std(axis=0, keepdims=True).shape == (1, 3)


def test_summary_keepdims():
    x = Symbol('x', '5 * 3 * float32')
    assert summary(a=x.min(), b=x.max()).dshape == \
            dshape('{a: float32, b: float32}')
    assert summary(a=x.min(), b=x.max(), keepdims=True).dshape == \
            dshape('1 * 1 * {a: float32, b: float32}')


def test_axis_kwarg_is_normalized_to_tuple():
    x = Symbol('x', '5 * 3 * float32')
    exprs = [x.sum(), x.sum(axis=1), x.sum(axis=[1]), x.std(), x.mean(axis=1)]
    for expr in exprs:
        assert isinstance(expr.axis, tuple)


def test_summary_with_multiple_children():
    t = Symbol('t', 'var * {x: int, y: int, z: int}')

    assert summary(a=t.x.sum() + t.y.sum())._child.isidentical(t)
