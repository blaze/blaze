from blaze import dshape, ones, zeros, open
from blaze.ts.ucr_dtw import ucr

def test_dtw():
    data  = ones(dshape('100, float32'))
    query = ones(dshape('100, float32'))

    loc, dist = ucr.dtw(data, query, 0.1, 100, verbose=False)

    # these are stupid, mostly just to check for regressions
    assert isinstance(loc, int)
    assert isinstance(dist, float)
