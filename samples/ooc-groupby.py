## Example of an implementation of an out-of-core groupby with Blaze.
## F. Alted
## 2013-10-10

"""
This script performs an out of core groupby operation for different datasets.

The datasets to be processed are normally in CSV files and the key and
value to be used for the grouping are defined programatically via small
functions (see toy_stream() and statsmodel_stream() for examples).

Those datasets included in statsmodel will require this package
installed (it is available in Anaconda, so it should be an easy
dependency to solve).

Usage: $ `script` dataset_class dataset_filename

`dataset_class` can be either 'toy', 'randhie' or 'contributions'.

'toy' is a self-contained dataset and is meant for debugging mainly.

The 'randhie' implements suport for the dataset with the same name
included in the statsmodel package.

Finally 'contributions' is meant to compute aggregations on the
contributions to the different US campaigns.  This latter requires a
second argument (datatset_filename) which is a CSV file downloaded from:
http://data.influenceexplorer.com/bulk/

"""

import sys
from itertools import islice
import io
import csv
from dynd import nd, ndt
import blz
import os.path
import numpy as np

# Number of lines to read per each iteration
LPC = 1000

# Max number of chars to map for a bytes or string in NumPy
MAXCHARS = 64

def get_nptype(dtype, val):
    """Convert the `val` field in dtype into a numpy dtype."""
    dytype = dtype[nd.as_py(dtype.field_names).index(val)]
    # strings and bytes cannot be natively represented in numpy
    if dytype == ndt.string:
        nptype = np.dtype("U%d" % MAXCHARS)
    elif dytype == ndt.bytes:
        nptype = np.dtype("S%d" % MAXCHARS)
    else:
        # There should be no problems with the rest
        nptype = dytype.as_numpy()
    return nptype


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

    try:
        nptype = get_nptype(dtype, val)
    except ValueError:
        raise ValueError("`val` should be a valid field")

    # Start reading chunks
    prev_keys = set()
    while True:
        ndbuf = nd.array(islice(sreader, lines_per_chunk), dtype)
        if len(ndbuf) == 0: break   # CSV data exhausted

        # Do the groupby for this chunk
        keys = getattr(ndbuf, key)
        if val is None:
            vals = ndbuf
        else:
            vals = getattr(ndbuf, val)
        sby = nd.groupby(vals, keys)
        lkeys = nd.as_py(sby.groups)
        skeys = set(lkeys)
        # BLZ does not understand dynd objects (yet)
        sby = nd.as_py(sby.eval())

        if len(prev_keys) == 0:
            # Add the initial keys to a BLZ table
            columns = [np.array(sby[i], nptype) for i in range(len(lkeys))]
            ssby = blz.btable(columns=columns, names=lkeys, rootdir=path,
                              mode='w')
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

        # Add the new keys to the existing ones
        prev_keys |= skeys

    # Before returning, flush all data into disk
    if path is not None:
        ssby.flush()
    return ssby


# A CSV toy example
csvbuf = u"""k1,v1,1,u1
k2,v2,2,u2
k3,v3,3,u3
k4,v4,4,u4
k5,v5,5,u5
k5,v6,6,u6
k4,v7,7,u7
k4,v8,8,u8
k4,v9,9,u9
k1,v10,10,u9
k5,v11,11,u11
"""

def toy_stream():
    sreader = csv.reader(io.StringIO(csvbuf))
    # The dynd dtype for the CSV file above
    dt = ndt.type('{key: string; val1: string; val2: int32; val3: bytes}')
    # The name of the persisted table where the groupby will be stored
    return sreader, dt


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
    return sreader, dtype

# For contributions to state and federal US campaings.
# CSV files can be downloaded from:
# http://data.influenceexplorer.com/bulk/
def contributions_stream(stream_file):
    f = open(stream_file, 'rb')
    # Description of this dataset
    headers = f.readline().strip()   # read out the headers line
    headers = headers.split(',')
    # The types for the different fields
    htypes = [ ndt.int32, ndt.int16, ndt.int16] + \
             [ ndt.string ] * 4 + \
             [ ndt.bool, ndt.float64 ] + \
             [ ndt.string ] * 33
    # Build the DyND data type
    dtype = ndt.make_struct(htypes, headers)
    sreader = csv.reader(f)
    return sreader, dtype


if __name__ == "__main__":

    # Which dataset do we want to group?
    which = sys.argv[1] if len(sys.argv) > 1 else "toy"

    if which == "toy":
        # Get the CSV iterator and dtype of fields
        sreader, dt = toy_stream()
        # Do the actual sortby
        ssby = groupby(sreader, 'key', 'val1', dtype=dt, path=None,
                       lines_per_chunk=2)
    elif which == "randhie":
        # Get the CSV iterator and dtype of fields
        sreader, dt = statsmodel_stream(which)
        # Do the actual sortby
        ssby = groupby(sreader, 'mdvis', 'lncoins', dtype=dt, path=None)
    elif which == "contributions":
        # Get the CSV iterator and dtype of fields
        stream_file = sys.argv[2]
        sreader, dt = contributions_stream(stream_file)
        # Do the actual sortby
        ssby = groupby(
            sreader, 'recipient_party', 'amount', dtype=dt, path='contribs.blz')
    else:
        raise NotImplementedError(
            "parsing for `%s` dataset not implemented" % which)

    # Retrieve the data in the BLZ structure
    #ssby = blz.open(path)  # open from disk, if ssby would be persistent
    for key in ssby.names:
        values = ssby[key]
        if which in ('toy', 'randhie'):
            print "key:", key, values
        elif which == 'contributions':
            print "Party: '%s'\tAmount: %13.2f\t#contribs: %8d" % \
                  (key, values.sum(), len(values))
