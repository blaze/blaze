'''Tests of blaze.metadata package.'''
import blaze

def test_metadata_has_prop():
    a = blaze.ones(blaze.dshape('20, 20, float64'))
    c = blaze.NDTable([(1.0, 1.0), (1.0, 1.0)], dshape='2, {x: int32; y: float32}')

    assert blaze.metadata.has_prop(a, blaze.metadata.arraylike)
    assert blaze.metadata.has_prop(c, blaze.metadata.tablelike)
    assert not blaze.metadata.has_prop(a, blaze.metadata.tablelike)

def test_metadata_all_prop():
    a = blaze.ones(blaze.dshape('20, 20, float64'))
    b = blaze.zeros(blaze.dshape('20, 20, float64'))
    c = blaze.NDTable([(1.0, 1.0), (1.0, 1.0)], dshape='2, {x: int32; y: float32}')

    assert blaze.metadata.all_prop((a, b), blaze.metadata.arraylike)
    assert not blaze.metadata.all_prop((a, b, c), blaze.metadata.arraylike)
