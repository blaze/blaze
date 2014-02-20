'''Sample module showing the creation of blaze arrays'''

from __future__ import absolute_import, division, print_function

import sys

import numpy as np
import blaze

try:
    import tables as tb
except ImportError:
    print("This example requires PyTables to run.")
    sys.exit()


def print_section(a_string, level=0):
    spacing = 2 if level == 0 else 1
    underline = ['=', '-', '~', ' '][min(level,3)]

    print ('%s%s\n%s' % ('\n'*spacing,
                         a_string,
                         underline*len(a_string)))

fname = "sample.h5"
print_section('building basic hdf5 files')
# Create a simple HDF5 file
a1 = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
a2 = np.array([[1, 2, 3], [3, 2, 1]], dtype="int64")
t1 = np.array([(1, 2, 3), (3, 2, 1)], dtype="i4,i8,f8")
with tb.open_file(fname, "w") as f:
    f.create_array(f.root, 'a1', a1)
    f.create_table(f.root, 't1', t1)
    f.create_group(f.root, 'g')
    f.create_array(f.root.g, 'a2', a2)
    print("Created HDF5 file with the next contents:\n%s" % str(f))

print_section('opening and handling datasets in hdf5 files')
# Open an homogeneous dataset there
store = blaze.Storage(fname, format='hdf5')
a = blaze.open(store, datapath="/a1")
# Print it
print("/a1 contents:", a)
# Print the datashape
print("datashape for /a1:", a.dshape)

# Open another homogeneous dataset there
store = blaze.Storage(fname, format='hdf5')
a = blaze.open(store, datapath="/g/a2")
# Print it
print("/g/a2 contents:", a)
# Print the datashape
print("datashape for /g/a2:", a.dshape)

# Now, get an heterogeneous dataset
store = blaze.Storage(fname, format='hdf5')
t = blaze.open(store, datapath="/t1")
# Print it
print("/t1 contents:", t)
# Print the datashape
print("datashape for /t1:", t.dshape)

# Finally, get rid of the sample file
blaze.drop(store)
