#------------------------------------------------------------------------
# CArray Namespace
#------------------------------------------------------------------------

# I give up fighting with circular deps, the original model of dumping
# most of the namespace in the __init__ doesn't appear to work when its
# a nested package. One would think that just changing all references of
# carray to blaze.carary would work but it doesn't unless you
# setup.py install it. :-(

# Print array functions (imported from NumPy)
from arrayprint import (
    array2string, set_printoptions, get_printoptions)

from carrayExtension import (
    carray,
    chunk,
    # blosc_version, _blosc_set_nthreads as blosc_set_nthreads
    )
from ctable import ctable
from toplevel import cparams, open, zeros, ones, fromiter
from version import __version__
