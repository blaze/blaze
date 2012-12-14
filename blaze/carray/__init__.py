#------------------------------------------------------------------------
# CArray Namespace
#------------------------------------------------------------------------

# Print array functions (imported from NumPy)
from arrayprint import (
    array2string, set_printoptions, get_printoptions)
from carrayExtension import (
    carray,
    chunk,
    # _cparams as cparams,
    # blosc_version, _blosc_set_nthreads as blosc_set_nthreads
    )
from toplevel import cparams, open, zeros, ones, fromiter
from version import __version__

# doesn't work for reasons I don't understand
#from ctable import ctable
