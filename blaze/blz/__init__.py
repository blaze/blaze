#------------------------------------------------------------------------
# BLZ Namespace
#------------------------------------------------------------------------

from __future__ import absolute_import

# Print array functions (copied from NumPy)
from .arrayprint import (
    array2string, set_printoptions, get_printoptions)

from .blz_ext import barray, blz_set_nthreads, blosc_version
from .btable import btable
from .bfuncs import (
    open, zeros, ones, fromiter, iterblocks, whereblocks)
from .bparams import bparams
from .version import __version__
