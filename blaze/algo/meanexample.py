import os

import blaze
from blaze.carray import carray
from blaze.carray.ctable import ctable

import numpy as np

STORAGE = 'example1'

#------------------------------------------------------------------------
if not os.path.exists(STORAGE):
    print 'Creating tables'
    N = 100000
    a = carray(np.arange(N, dtype='i4'))
    b = carray(np.arange(N, dtype='f8')+1)
    t = ctable((a, b), ('f0', 'f1'), rootdir='example1', mode='w')
    t.flush()
#------------------------------------------------------------------------

from time import time

print '-------------------'

t = blaze.open('ctable://example1')

# Using chunked blaze array we can optimize for IO doing the sum
# operations chunkwise from disk.

t0 = time()
print blaze.mean(t, 'f0')
print "Chunked mean", round(time()-t0, 6)

# Using NumPy is just going to through the iterator protocol on
# carray which isn't going to efficient.

t0 = time()
print np.mean(t.data.ca['f0'])
print "NumPy mean", round(time()-t0, 6)

print '==================='

t0 = time()
#assert blaze.std(t, 'f0') == 28867.513458037913
print blaze.std(t, 'f0')
print "Chunked std", round(time()-t0, 6)

print '-------------------'

t0 = time()
print np.std(t.data.ca['f0'])
print "NumPy std", round(time()-t0, 6)

#blaze.generic_loop(t, 'f0')
