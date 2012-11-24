__version__ = '0.1-dev'

try:
    from ndtable.datashape import dshape
    from ndtable.table import Array, Table, NDArray, NDTable
except ImportError as e:
    # prevent weird cyclic import errors from passing silently
    raise RuntimeError("Failed to load, check for cyclic imports.")

# From numpy compatability, ideally ``import ndtable as np``
# should be somewhat backwards compatable
array   = Array
ndarray = NDArray
dtype   = dshape
