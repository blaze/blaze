## Example of an implementation of an out-of-core groupby for BLZ
## F. Alted
## 2013-10-10

from itertools import islice
import io
import csv
from dynd import nd, ndt
from blaze import blz
import os.path
from shutil import rmtree
import numpy as np

# Number of lines to read per each iteration
LPC = 2

# Max number of chars to map for a bytes or string in NumPy
MAXCHARS = 64

def groupby(sreader, key, val, dtype, path=None, lines_per_chunk=LPC):
    """Group the `val` field in `sreader` stream of lines by `key` index.

    Parameters
    ----------
    sreader : iterator
        Iterator over a stream of CSV lines.
    key : string
        The name of the field to be grouped by.
    val : string
        The field name with the values that have to be grouped.
    dtype : dynd dtype
        The DyND data type with all the fields of the CSV lines,
        including the `key` and `val` names.
    path : string
        The path of the file where the BLZ array with the final
        grouping will be stored.  If None (default), the BLZ will be
        stored in-memory (and hence non-persistent).
    lines_per_chunk : int
        The number of chunks that have to be read to be grouped by
        in-memory.  For optimal perfomance, some experimentation
        should be needed.  The default value should work reasonably
        well, though.
        
    Returns
    -------
    output : BLZ table
        Returns a BLZ table with column names that are the groups
        resulting from the groupby operation.  The columns are filled
        with the `val` field of the lines delivered by `sreader`.

    """

    # Convert the `val` field into a numpy dtype
    dytype = dtype[nd.as_py(dtype.field_names).index('val')]
    # strings and bytes cannot be natively represented in numpy
    if dytype == ndt.string:
        nptype = "U%d" % MAXCHARS
    elif dytype == ndt.bytes:
        nptype = "S%d" % MAXCHARS
    else:
        # There should be no problems with the rest
        nptype = dydtype.as_numpy()
    
    # Start reading chunks
    prev_keys = set()
    while True:
        a = nd.array(islice(sreader, lines_per_chunk), dtype)
        if len(a) == 0: break   # CSV data exhausted

        # Get the set of keys for this chunk
        keys = nd.as_py(getattr(a, key))
        skeys = set(keys)
        keys = list(skeys)
        
        # Do the groupby for this chunk
        sby = nd.groupby(getattr(a, val), getattr(a, key), keys).eval()
        sby = nd.as_py(sby)

        if len(prev_keys) == 0:
            # Check path and if it exists, remove it and every
            # directory below it
            if os.path.exists(path): rmtree(path)
            # Add the initial keys to a BLZ table
            columns = [np.array(sby[i], nptype) for i in range(len(keys))]
            ssby = blz.btable(columns=columns, names=keys, rootdir=path)
        else:
            # Have we new keys?
            new_keys = skeys.difference(prev_keys)
            for new_key in new_keys:
                # Get the index of the new key
                idx = keys.index(new_key)
                # and add the values as a new columns
                ssby.addcol(sby[idx], new_key, dtype=nptype)
            # Now fill the pre-existing keys
            existing_keys = skeys.intersection(prev_keys)
            for existing_key in existing_keys:
                # Get the index of the existing key
                idx = keys.index(existing_key)
                # and append the values here
                ssby[existing_key].append(sby[idx])
            assert skeys == existing_keys | new_keys

        # Add the new keys to the existing ones
        prev_keys |= skeys

    return ssby


# A CSV toy example
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

if __name__ == "__main__":
    
    # The iterator for reading the CSV file line by line
    sreader = csv.reader(io.StringIO(csvbuf))
    
    # The dynd dtype for the CSV file above
    dt = ndt.type('{key: string; val: string}')
    
    # The name of the persisted table where the groupby will be stored
    path = 'persisted.blz'

    # Do the actual sortby
    ssby = groupby(sreader, 'key', 'val', dtype=dt, path=path)
    
    # Finally, print the ssby table (do not try to dump it in the
    # traditional way because the length of the columns is not the same)
    # print "ssby:", ssby
    for key in ssby.names:
        print "key:", key, ssby[key]
