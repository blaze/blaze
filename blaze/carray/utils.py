########################################################################
#
#       License: BSD
#       Created: August 5, 2010
#       Author:  Francesc Alted - francesc@continuum.io
#
########################################################################

"""Utility functions (mostly private).
"""

import sys, os, os.path, subprocess, math
from time import time, clock
import numpy as np


def show_stats(explain, tref):
    "Show the used memory (only works for Linux 2.6.x)."
    # Build the command to obtain memory info
    cmd = "cat /proc/%s/status" % os.getpid()
    sout = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    for line in sout:
        if line.startswith("VmSize:"):
            vmsize = int(line.split()[1])
        elif line.startswith("VmRSS:"):
            vmrss = int(line.split()[1])
        elif line.startswith("VmData:"):
            vmdata = int(line.split()[1])
        elif line.startswith("VmStk:"):
            vmstk = int(line.split()[1])
        elif line.startswith("VmExe:"):
            vmexe = int(line.split()[1])
        elif line.startswith("VmLib:"):
            vmlib = int(line.split()[1])
    sout.close()
    print "Memory usage: ******* %s *******" % explain
    print "VmSize: %7s kB\tVmRSS: %7s kB" % (vmsize, vmrss)
    print "VmData: %7s kB\tVmStk: %7s kB" % (vmdata, vmstk)
    print "VmExe:  %7s kB\tVmLib: %7s kB" % (vmexe, vmlib)
    tnow = time()
    print "WallClock time:", round(tnow - tref, 3)
    return tnow


##### Code for computing optimum chunksize follows  #####

def csformula(expectedsizeinMB):
    """Return the fitted chunksize for expectedsizeinMB."""
    # For a basesize of 1 KB, this will return:
    # 4 KB for datasets <= .1 KB
    # 64 KB for datasets == 1 MB
    # 1 MB for datasets >= 10 GB
    basesize = 1024
    return basesize * int(2**(math.log10(expectedsizeinMB)+6))

def limit_es(expectedsizeinMB):
    """Protection against creating too small or too large chunks."""
    if expectedsizeinMB < 1e-4:     # < .1 KB
        expectedsizeinMB = 1e-4
    elif expectedsizeinMB > 1e4:    # > 10 GB
        expectedsizeinMB = 1e4
    return expectedsizeinMB

def calc_chunksize(expectedsizeinMB):
    """Compute the optimum chunksize for memory I/O in carray/ctable.

    carray stores the data in chunks and there is an optimal length for
    this chunk for compression purposes (it is around 1 MB for modern
    processors).  However, due to the implementation, carray logic needs
    to always reserve all this space in-memory.  Booking 1 MB is not a
    drawback for large carrays (>> 1 MB), but for smaller ones this is
    too much overhead.

    The tuning of the chunksize parameter affects the performance and
    the memory consumed.  This is based on my own experiments and, as
    always, your mileage may vary.
    """

    expectedsizeinMB = limit_es(expectedsizeinMB)
    zone = int(math.log10(expectedsizeinMB))
    expectedsizeinMB = 10**zone
    chunksize = csformula(expectedsizeinMB)
    return chunksize

def get_len_of_range(start, stop, step):
    """Get the length of a (start, stop, step) range."""
    n = 0
    if start < stop:
        n = ((stop - start - 1) // step + 1);
    return n

def to_ndarray(array, dtype, arrlen=None):
    """Convert object to a ndarray."""

    if dtype is None:
        return np.array(array)

    # Arrays with a 0 stride are special
    if type(array) == np.ndarray and array.strides[0] == 0:
        if array.dtype != dtype.base:
            raise TypeError, "dtypes do not match"
        return array

    # Ensure that we have an ndarray of the correct dtype
    if type(array) != np.ndarray or array.dtype != dtype.base:
        try:
            array = np.array(array, dtype=dtype.base)
        except ValueError:
            raise ValueError, "cannot convert to an ndarray object"

    # We need a contiguous array
    if not array.flags.contiguous:
        array = array.copy()
    if len(array.shape) == 0:
        # We treat scalars like undimensional arrays
        array.shape = (1,)

    # Check if we need a broadcast
    if arrlen is not None and arrlen != len(array):
        array2 = np.empty(shape=(arrlen,), dtype=dtype)
        array2[:] = array   # broadcast
        array = array2

    print "array:", `array`
    return array

def human_readable_size(size):
    """Return a string for better assessing large number of bytes."""
    if size < 2**10:
        return "%s" % size
    elif size < 2**20:
        return "%.2f KB" % (size / float(2**10))
    elif size < 2**30:
        return "%.2f MB" % (size / float(2**20))
    elif size < 2**40:
        return "%.2f GB" % (size / float(2**30))
    else:
        return "%.2f TB" % (size / float(2**40))


# Main part
# =========
if __name__ == '__main__':
    print human_readable_size(1023)
    print human_readable_size(10234)
    print human_readable_size(10234*100)
    print human_readable_size(10234*10000)
    print human_readable_size(10234*1000000)
    print human_readable_size(10234*100000000)
    print human_readable_size(10234*1000000000)


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
