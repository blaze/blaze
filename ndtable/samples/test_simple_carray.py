# Deliverable #1

# Simple use case for connecting CArray API and Blaze
# Byte Providers.

from ndtable.table import Array

#------------------------------------------------------------------------
# Case
#------------------------------------------------------------------------

# We use the Array object which is immediete in all operations.

def arr():
    return Array([1,2,3], datashape='3, int32')

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

# For now, just assert that these operations are implemented.
# We'll implement equality check letter and add assertions that
# check the end result.

# Case 0
def test_getitem():
    nd = arr()

    for i in xrange(3):
        # Simple read, forces the manifest bytes
        print nd[i]

# Case 1
def test_setitem():
    nd = arr()

    for i in xrange(3):
        # Simple write, forces the manifest bytes
        nd[i] = i

# Case 2
def test_getslice():
    nd = arr()

    for i in xrange(3):
        # Simple read slice
        nd[0:i] = i

# Case 3
def test_setslice():
    nd = arr()

    for i in xrange(3):
        # Simple write slice
        nd[0:i] = i

# Case 4
def test_fancyslice():
    nd = arr()

    for i in xrange(3):
        # Simple read fancy slice
        nd[i::2]
