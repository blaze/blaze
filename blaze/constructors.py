from __future__ import absolute_import

# This are the constructors for the blaze array objects.  Having them
# as external functions allows to more flexibility and helps keeping
# the blaze array object compact, just showing the interface of the
# array itself.
#
# The blaze array __init__ method should be considered private and for
# advanced users only. It will provide the tools supporting the rest
# of the constructors, and will use low-level parameters, like
# ByteProviders, that an end user may not even need to know about.

from .array import Array
from .datadescriptor import NumPyDataDescriptor, BLZDataDescriptor
from .datashape import to_numpy, dshape

# note that this is rather naive. In fact, a proper way to implement
# the array from a numpy is creating a ByteProvider based on "data"
# and infer the indexer from the apropriate information in the numpy
# array.
def array(obj, dshape=None, caps={'efficient-write': True}):
    """Create an in-memory Blaze array.

    Parameters
    ----------
    data : array_lile
        Initial contents for the array.

    dshape : datashape
        The datashape for the resulting array. By default the
        datashape will be inferred from data. If an explicit dshape is
        provided, the input data will be coerced into the provided
        dshape.

	caps : capabilities dictionary
	    A dictionary containing the desired capabilities of the array

    Returns
    -------
    out : a concrete, in-memory blaze array.

    Bugs
    ----
    Right now the explicit dshape is ignored. This needs to be
    corrected. When the data cannot be coerced to an explicit dshape
    an exception should be raised.

    """
    if 'efficient-write' in caps:
        # NumPy provides efficient writes
        import numpy as np
        dd = NumPyDataDescriptor(np.array(obj))
    elif 'compress' in caps:
        # BLZ provides compression
        dd = BLZDataDescriptor(obj)
    return Array(dd)

    datadesc = NumPyDataDescriptor(array(data))

    return Array(datadesc)


def zeros(ds):
    """Create an array and fill it with zeros

    Parameters
    ----------
    ds : datashape
        The datashape for the created array.

    Returns
    -------
    out: a concrete blaze array

    Bugs
    ----
    Right now only concrete, in-memory blaze arrays can be created
    this way.

    """
    from numpy import zeros

    ds = ds if not isinstance(ds, basestring) else dshape(ds)
    (shape, dtype) = to_numpy(ds)
    datadesc = NumPyDataDescriptor(zeros(shape, dtype=dtype))
    return Array(datadesc)


def ones(ds):
    """Create an array and fill it with ones

    Parameters
    ----------
    ds : datashape
        The datashape for the created array.

    Returns
    -------
    out: a concrete blaze array

    Bugs
    ----
    Right now only concrete, in-memory blaze arrays can be created
    this way.

    """
    from numpy import ones

    ds = ds if not isinstance(ds, basestring) else dshape(ds)
    (shape, dtype) = to_numpy(ds)
    datadesc = NumPyDataDescriptor(ones(shape, dtype=dtype))
    return Array(datadesc)


# for a temptative open function:
def open(uri):
    raise NotImplementedError
