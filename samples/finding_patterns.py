"""
Script for performing queries on large time series by using UCR ED and
DTW algs.

The computation is performed with completely on-disk datasets.  After
the run a couple of datasets are created in the next directories:

ts.blz: the time series (just a simple function like x*sin(x))
query.blz: an exact pattern to be found
query2.blz: a noise dataset to be found

"""

import os, shutil, math
from time import time
import blaze as blz
from blaze.ts.ucr_dtw import ucr
import numpy as np

NROWS = int(1e6)
NPERIODS = int(1e4)
CS = NROWS / NPERIODS
timing = True

t0 = time()
# Create a large time series dataset:
if os.path.exists('ts.blz'): shutil.rmtree('ts.blz')
ts = blz.array([], 'x, float64', params=blz.params(storage='ts.blz'))
for i in range(NPERIODS):
    # Proceed to fill the empty array in chunks
    x = np.linspace(i*math.pi, (i+1)*math.pi, CS)
    ts.append(x*np.sin(x))
ts.commit()

# Create a dataset to query
if os.path.exists('query.blz'): shutil.rmtree('query.blz')
xq = np.linspace(3*math.pi, 4*math.pi, CS)
query = blz.array(xq*np.sin(xq), params=blz.params(storage='query.blz'))
if os.path.exists('query2.blz'): shutil.rmtree('query2.blz')
n = np.random.randn(query.size)*.1  # introduce some noise
query2 = blz.array(xq*np.sin(xq)+n, params=blz.params(storage='query2.blz'))
if timing: print "Total Blaze arrays create time :", round(time()-t0, 4)

t0 = time()
# Open Blaze arrays on-disk (will not be loaded in memory)
ts = blz.open("ts.blz")
query = blz.open("query.blz")
query2 = blz.open("query2.blz")
if timing: print "Total Blaze arrays open time :", round(time()-t0, 4)
print "query size:", query.size

# Do the search for the exact pattern
print "   ***   Querying *exact* pattern   ***"
t0 = time()
loc, dist = ucr.dtw(ts, query, 0.1, query.size, verbose=False)
print "Location : ", loc
print "Distance : ", dist
print "Data Scanned : ", ts.size
if timing: print "Total Execution Time (exact):", round(time()-t0, 4)

# Do the search for the noisy pattern
print "   ***   Querying *noisy* pattern   ***"
t0 = time()
loc, dist = ucr.dtw(ts, query2, 0.1, query2.size, verbose=False)
print "Location : ", loc
print "Distance : ", dist
print "Data Scanned : ", ts.size
if timing: print "Total Execution Time (noisy) :", round(time()-t0, 4)
