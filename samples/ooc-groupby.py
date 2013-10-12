## Example of an implementation of an out-of-core groupby for BLZ
## F. Alted
## 2013-10-10

"""
This script performs an out of core groupby operation for different datasets.

The datasets to be processed are normally in CSV files and the key and
values to be used in the grouped are defined programatically via small
functions (see toy_stream() and statsmodel_stream() for examples).

Those datasets included in statsmodel will require this package
installed (it is available in Anaconda, so it should be an easy
dependency to solve).

Usage: `script` dataset_name
"""

from itertools import islice
import io
import csv
from dynd import nd, ndt
from blaze import blz
import os.path
from shutil import rmtree
import numpy as np

# Number of lines to read per each iteration
LPC = 100

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
    dytype = dtype[nd.as_py(dtype.field_names).index(val)]
    # strings and bytes cannot be natively represented in numpy
    if dytype == ndt.string:
        nptype = "U%d" % MAXCHARS
    elif dytype == ndt.bytes:
        nptype = "S%d" % MAXCHARS
    else:
        # There should be no problems with the rest
        nptype = dytype.as_numpy()
    
    # Start reading chunks
    prev_keys = set()
    while True:
        ndbuf = nd.array(islice(sreader, lines_per_chunk), dtype)
        if len(ndbuf) == 0: break   # CSV data exhausted

        # Do the groupby for this chunk
        keys = getattr(ndbuf, key)
        lkeys = nd.as_py(keys)
        skeys = set(lkeys)
        lkeys = list(skeys)
        vals = getattr(ndbuf, val)
        sby = nd.groupby(vals, keys, lkeys)
        sby = nd.as_py(sby.eval())

        if len(prev_keys) == 0:
            # Check path and if it exists, remove it and every
            # directory below it
            if os.path.exists(path): rmtree(path)
            # Add the initial keys to a BLZ table
            columns = [np.array(sby[i], nptype) for i in range(len(lkeys))]
            ssby = blz.btable(columns=columns, names=lkeys, rootdir=path)
        else:
            # Have we new keys?
            new_keys = skeys.difference(prev_keys)
            for new_key in new_keys:
                # Get the index of the new key
                idx = lkeys.index(new_key)
                # and add the values as a new columns
                ssby.addcol(sby[idx], new_key, dtype=nptype)
            # Now fill the pre-existing keys
            existing_keys = skeys.intersection(prev_keys)
            for existing_key in existing_keys:
                # Get the index of the existing key
                idx = lkeys.index(existing_key)
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

def toy_stream():
    sreader = csv.reader(io.StringIO(csvbuf))
    # The dynd dtype for the CSV file above
    dt = ndt.type('{key: string; val: string}')
    # The name of the persisted table where the groupby will be stored
    path = 'toy.blz'
    return sreader, dt, path


# This access different datasets in statsmodel package
def statsmodel_stream(stream):
    import statsmodels.api as sm
    data = getattr(sm.datasets, stream)
    f = open(data.PATH, 'rb')
    if stream == 'randhie':
        # For a description of this dataset, see:
        # http://statsmodels.sourceforge.net/devel/datasets/generated/randhie.html
        f.readline()   # read out the headers line
        dtypes = ('{mdvis: string; lncoins: float32; idp: int32;'
                  ' lpi:float32; fmde: float32; physlm: float32;'
                  ' disea: float32; hlthg: int32; hlthf: int32;'
                  ' hlthp: int32}')
    else:
        raise NotImplementedError(
            "Importing this dataset has not been implemented yet")

    sreader = csv.reader(f)
    dtype = ndt.type(dtypes)
    return sreader, dtype, stream+".blz"


if __name__ == "__main__":
    import sys

    # Which dataset do we want to group?
    which = sys.argv[1] if len(sys.argv) > 1 else "toy"

    if which == "toy":
        # The iterator for reading the toy CSV file line by line
        sreader, dt, path = toy_stream()
        # Do the actual sortby
        ssby = groupby(sreader, 'key', 'val', dtype=dt, path=path,
                       lines_per_chunk=2)
    elif which == "randhie":
        # The iterator and dtype for datasets included in statsmodel
        sreader, dt, path = statsmodel_stream(which)
        # Do the actual sortby
        ssby = groupby(sreader, 'mdvis', 'lncoins', dtype=dt, path=path)
    else:
        raise ValueError(
            "parsing for `%s` dataset not implemented"
            "(try either 'toy' or 'randhie')" % which)

    ssby.flush()   # flush all the data in blz object

    # Reopen the BLZ object on-disk for retrieving the grouped data
    ssby = blz.open(path)
    # Finally, print the ssby table (do not try to dump it in the
    # traditional way because the length of the columns is not the same)
    # print "ssby:", ssby
    names = ssby.names[:]
    # Additional sort for guaranteeing sorted keys too
    names.sort()
    for key in names:
        print "key:", key, ssby[key]
