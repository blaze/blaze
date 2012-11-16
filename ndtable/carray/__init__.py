#------------------------------------------------------------------------
# CArray Namespace
#------------------------------------------------------------------------

from carray.arrayprint import (
   array2string, set_printoptions, get_printoptions)

from carray.carrayExtension import (
   carray, blosc_version,
   _blosc_set_nthreads as blosc_set_nthreads
)
from carray.ctable import ctable
from carray.toplevel import (
   detect_number_of_cores, set_nthreads,
   open, fromiter, arange, zeros, ones, fill,
   cparams, eval, walk )
from carray.version import __version__
from defaults import defaults
