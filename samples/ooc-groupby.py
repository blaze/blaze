## Example of an implementation of an out-of-core groupby for BLZ
## F. Alted
## 2013-10-10

from itertools import islice
import io
import csv
from dynd import nd, ndt
from blaze import blz


# The toy CSV example
csvbuf = u"""k1,v1
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

# Number of rows to read per each iteration
nrows_in_chunk = 2

# The dynd dtype for the CSV file above
dt = ndt.type('{key: string; val: string}')

# The iterator for reading the CSV file
reader = csv.reader(io.StringIO(csvbuf))

# Start reading chunks
prev_keys = set()
while True:
    a = nd.array(islice(reader, nrows_in_chunk), dt)
    if len(a) == 0: break   # CSV data exhausted

    keys = nd.as_py(a.key)
    skeys = set(keys)
    keys = list(skeys)

    # Do the groupby for this chunk using dynd
    sby = nd.groupby(a.val, a.key, keys).eval()
    sby = nd.as_py(sby)

    # Add the initial keys to a BLZ table
    if len(prev_keys) == 0:
        ssby = blz.btable(columns=sby, names=keys)
    else:
        # Have we new keys?
        new_keys = skeys.difference(prev_keys)
        for new_key in new_keys:
            # Get the index of the new key
            idx = keys.index(new_key)
            # and add the values as a new columns
            ssby.addcol(sby[idx], new_key)
        # Now fill the pre-existing keys
        existing_keys = skeys.intersection(prev_keys)
        for existing_key in existing_keys:
            # Get the index of the existing key
            idx = keys.index(existing_key)
            # and append the values here
            ssby[existing_key].append(sby[idx])

    prev_keys |= skeys

# Finally, print the result (do not try to dump it in the traditional
# way because the length of the columns is not the same)
# print "ssby:", ssby
for key in prev_keys:
    print "key:", key, ssby[key]
