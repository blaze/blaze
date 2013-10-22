## Example of an implementation of an out-of-core groupby with DyND and BLZ
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

Usage: $ `script` dataset_class dataset_filename

`dataset_class` can be either 'toy', 'randhie' or 'contributions'.
The 'toy' is a self-contained dataset and is meant for debugging
mainly.  The 'randhie' implements suport for the dataset with the same
name included in the statsmodel package.  Finally 'contributions' is
meant to compute aggregations on the contributions to the different US
campaigns.  This latter requires a second argument (datatset_filename)
which is a CSV file downloaded from:
http://data.influenceexplorer.com/bulk/

"""

import sys
from itertools import islice
import io
import csv
from dynd import nd, ndt
from blaze import blz
import os.path
import numpy as np

# Number of lines to read per each iteration
LPC = 1000

# Max number of chars to map for a bytes or string in NumPy
MAXCHARS = 64

def get_nptype(dtype, val):
    # Convert the `val` field into a numpy dtype
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

    if val is None:
        types = [(bytes(name), get_nptype(dtype, name))
                 for name in nd.as_py(dtype.field_names)]
        nptype = np.dtype(types)
    else:
        nptype = get_nptype(dtype, val)

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
            assert skeys == existing_keys | new_keys

        # Add the new keys to the existing ones
        prev_keys |= skeys

    # Before returning, flush all data into disk
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
    return sreader, dt, 'toy.blz'


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

# For contributions to state and federal US campaings.
# CSV files can be downloaded from:
# http://data.influenceexplorer.com/bulk/
def contributions_stream(stream_file):
    f = open(stream_file, 'rb')
    # Description of this dataset
    headers = f.readline().strip()   # read out the headers line
    headers = headers.split(',')
    htypes = [
        ndt.int32,    # id
        ndt.int16,    # import_reference_id
        ndt.int16,    # cycle (year)
        ndt.string,   # transaction_namespace
        ndt.string,   # transaction_id
        ndt.string,   # transaction_type
        ndt.string,   # filing_id
        ndt.bool,     # is_amendment
        ndt.float64,  # amount
        ndt.string,   # date
        ndt.string,   # contributor_name
        ndt.string,   # contributor_ext_id
        ndt.string,   # contributor_type
        ndt.string,   # contributor_occupation
        ndt.string,   # contributor_employer
        ndt.string,   # contributor_gender
        ndt.string,   # contributor_address
        ndt.string,   # contributor_city
        ndt.string,   # contributor_state
        ndt.string,   # contributor_zipcode
        ndt.string,   # contributor_category
        ndt.string,   # organization_name
        ndt.string,   # organization_ext_id
        ndt.string,   # parent_organization_name
        ndt.string,   # parent_organization_ext_id
        ndt.string,   # recipient_name
        ndt.string,   # recipient_ext_id
        ndt.string,   # recipient_party
        ndt.string,   # recipient_type
        ndt.string,   # recipient_state
        ndt.string,   # recipient_state_held
        ndt.string,   # recipient_category
        ndt.string,   # committee_name
        ndt.string,   # committee_ext_id
        ndt.string,   # committee_party
        ndt.bool,     # candidacy_status
        ndt.string,   # district
        ndt.string,   # district_held
        ndt.string,   # seat
        ndt.string,   # seat_held
        ndt.string,   # seat_status
        ndt.string,   # seat_result
        ]

    dtype = ndt.make_struct(htypes, headers)
    sreader = csv.reader(f)
    return sreader, dtype, "contributions.blz"


if __name__ == "__main__":

    # Which dataset do we want to group?
    which = sys.argv[1] if len(sys.argv) > 1 else "toy"

    if which == "toy":
        # The iterator for reading the toy CSV file line by line
        sreader, dt, path = toy_stream()
        # Do the actual sortby
        ssby = groupby(sreader, 'key', 'val1', dtype=dt, path=path,
                       lines_per_chunk=2)
    elif which == "randhie":
        # The iterator and dtype for datasets included in statsmodel
        sreader, dt, path = statsmodel_stream(which)
        # Do the actual sortby
        ssby = groupby(sreader, 'mdvis', 'lncoins', dtype=dt, path=path)
    elif which == "contributions":
        # The iterator and dtype for datasets included in statsmodel
        stream_file = sys.argv[2]
        sreader, dt, path = contributions_stream(stream_file)
        # Do the actual sortby
        ssby = groupby(sreader, 'recipient_party', 'amount',
                       dtype=dt, path=path)
    else:
        raise NotImplementedError(
            "parsing for `%s` dataset not implemented" % which)

    # Reopen the BLZ object on-disk for retrieving the grouped data
    ssby = blz.open(path)
    for key in ssby.names:
        values = ssby[key]
        if which in ('toy', 'randhie'):
            print "key:", key, values
        elif which == 'contributions':
            print "Party: '%s'\tAmount: %13.2f\t#contribs: %8d" % \
                  (key, values.sum(), len(values))
