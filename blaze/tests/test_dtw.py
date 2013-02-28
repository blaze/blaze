
from blaze import dshape, fromiter
from blaze.ts.ucr_dtw import ucr
from math import sin, pi
from nose.tools import ok_, eq_

def test_dtw():
    # note: ucr.dtw only supports float64 atm
    count = 100 
    data  = fromiter((sin(2*pi*i/count) for i in xrange(count)), 'x, float64')
    query = data[50:60]

    loc, dist = ucr.dtw(data, query, 0.1, verbose=False)
    
    # these are stupid, mostly just to check for regressions
    ok_ (isinstance(loc, (int, long)))
    ok_ (isinstance(dist, float))
    eq_ (loc, 50)
    ok_ (dist < 1e-10 and dist >= 0.0)


## Local Variables:
## mode: python
## coding: utf-8 
## python-indent: 4
## tab-width: 4
## fill-column: 66
## End:
