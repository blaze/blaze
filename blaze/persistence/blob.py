import numpy as np
import blaze
from blaze.carray import carray, cparams

# Create a new array
arrobj = np.array(["s%d"%i for i in range(10)], dtype='O')
# arrobj = []

blb = carray(arrobj, dtype='O', rootdir='p', mode='w')

# # Append some objects
# for i in xrange(10):
#     blb.append("s%d"%i)

#print "blb:", blb
blb.flush()

# Reopen the old array
blb = carray(rootdir='p')

# Get the objects out
for i in xrange(10):
    print blb[i]
