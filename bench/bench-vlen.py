import os.path
import shutil
"""
Benchmark that compares the storing of objects in both Blaze and PyTables
"""

from time import time
import blaze
import tables

N = 500

if os.path.exists('c'):
    shutil.rmtree('c')

t0 = time()
c = blaze.Array([], 'x, object', params=blaze.params(storage='c', clevel=5))

for i in xrange(N):
    c.append(u"s"*N*i)
c.commit()
print "time taken for writing in Blaze: %.3f" % (time() - t0)

t0 = time()
c2 = blaze.open('c')
#c2 = c
#print c2.datashape

tlen = 0
for i in range(N):
    #print "i:", i, repr(c2[i]), type(c2[i])
    tlen += len(c2[i][()])
print "time taken for reading in Blaze: %.3f" % (time() - t0)
print "tlen", tlen


# Create a VLArray:
t0 = time()
f = tables.openFile('vlarray.h5', mode='w')
vlarray = f.createVLArray(f.root, 'vlarray',
                          tables.ObjectAtom(),
                          "array of objects",
                          filters=tables.Filters(5))

for i in xrange(N):
    vlarray.append(u"s"*N*i)
f.close()
print "time taken for writing in HDF5: %.3f" % (time() - t0)

# Read the VLArray:
t0 = time()
f = tables.openFile('vlarray.h5', mode='r')
vlarray = f.root.vlarray

tlen = 0
for obj in vlarray:
    tlen += len(obj)
f.close()
print "time taken for reading in HDF5: %.3f" % (time() - t0)
print "tlen", tlen

