#------------------------------------------------------------------------
# CArray Namespace
#------------------------------------------------------------------------

# Print array functions (imported from NumPy)
from arrayprint import (
    array2string, set_printoptions, get_printoptions)
from carrayExtension import (
    carray, blosc_version, _blosc_set_nthreads as blosc_set_nthreads )
from version import __version__
