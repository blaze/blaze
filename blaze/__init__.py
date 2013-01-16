__version__ = '0.1-dev'

# Install the Blaze library of dispatch functions, must be called
# early
from lib import *

from datashape import dshape
from table import Array, Table, NDArray, NDTable

from eclass import eclass
manifest = eclass.manifest
delayed = eclass.delayed

# From numpy compatability, ideally ``import blaze as np``
# should be somewhat backwards compatable
array   = Array
ndarray = NDArray
dtype   = dshape

from params import params
from toplevel import open, zeros, ones, fromiter

# Shorthand namespace dump
from datashape.shorthand import *

# Record class declarations
from datashape.record import RecordDecl, derived

# The compatability wrappers
from datashape.coretypes import to_numpy, from_numpy

# Errors
from error import *

# For Ilan
from blaze.testing import runner as test
