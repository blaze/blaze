__version__ = '0.1-dev'

#try:
from datashape import dshape
from table import Array, Table, NDArray, NDTable
#except ImportError as e:
    ## prevent weird cyclic import errors from passing silently
    #raise ImportError, "Failed to load, check for cyclic imports ( %s )"%\
        #e.message

# From numpy compatability, ideally ``import blaze as np``
# should be somewhat backwards compatable
array   = Array
ndarray = NDArray
dtype   = dshape

# Shorthand namespace dump
from datashape.shorthand import *

# Record class declarations
from datashape.record import RecordDecl, derived

from blaze.error import *