import io
import csv
from dynd import nd, ndt
from blaze import blz
import numpy as np

csvbuf = u"""k1, v1
k2,v2
k3,v3
k4,v4
k5,v5
k5,v6
k4,v7
k4,v8
k4,v9
k1,v10
k5,v11
"""

# csvfile = io.StringIO(csvbuf)
# for row in csv.reader(csvfile):
#     print "row:", row[0], row[1]

reader = csv.reader(io.StringIO(csvbuf))

dt = ndt.type('{key: string; val: string}')
a = nd.array(reader, dt)

print "a:", repr(a)

keys = nd.as_py(a.key)
keys = list(set(keys))
keys.sort()
print "keys:", keys

sby = nd.groupby(a.val, a.key, keys).eval()
sby = nd.as_py(sby)
print "sortby:", sby

ssby = blz.btable(columns=sby, names=keys)
#print "ssby:", ssby
for key in keys:
    print "key:", key, ssby[key]
