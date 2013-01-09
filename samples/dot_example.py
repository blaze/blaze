# Script for benchmarking OOC matrix matrix multiplication (only 2D supported)

import shutil, os.path
from time import time
import blaze

from blaze.algo.linalg import dot

# Remove pre-existent data directories
for d in ('a', 'b', 'out'):
    if os.path.exists(d):
        shutil.rmtree(d)

# Create simple inputs
t0 = time()
a = blaze.ones(blaze.dshape('2000, 2000, float64'),
               params=blaze.params(storage='a'))
print "Time for matrix a creation : ", round(time()-t0, 3)
t0 = time()
b = blaze.ones(blaze.dshape('2000, 3000, float64'),
               params=blaze.params(storage='b'))
print "Time for matrix b creation : ", round(time()-t0, 3)

# Do the dot product
t0 = time()
out = dot(a, b, outname='out')
print "Time for ooc matmul : ", round(time()-t0, 3)
print "out:", out
