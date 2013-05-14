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
from .datadescriptor import (IDataDescriptor,
                NumPyDataDescriptor, BLZDataDescriptor)
from .datashape import dshape as _dshape_builder, to_numpy, to_dtype

import numpy as np
from . import blz

# note that this is rather naive. In fact, a proper way to implement
# the array from a numpy is creating a ByteProvider based on "obj"
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
    dshape = dshape if not isinstance(dshape, basestring) else _dshape_builder(dshape)

    if isinstance(obj, IDataDescriptor):
        # TODO: Validate the 'caps', convert to another kind
        #       of data descriptor if necessary
        dd = obj
    elif isinstance(obj, np.ndarray):
        dd = NumPyDataDescriptor(obj)
    elif isinstance(obj, blz.barray):
        dd = BLZDataDescriptor(obj)
    elif 'efficient-write' in caps:
        dt = None if dshape is None else to_dtype(dshape)
            
        # NumPy provides efficient writes
        dd = NumPyDataDescriptor(np.array(obj, dtype=dt))
    elif 'compress' in caps:
        # BLZ provides compression
        dd = BLZDataDescriptor(blz.barray(obj))
    else:
        raise TypeError(('Failed to construct blaze array from '
                        'object of type %r') % type(obj))
    return Array(dd)


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

    ds = ds if not isinstance(ds, basestring) else _dshape_builder(ds)
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

    ds = ds if not isinstance(ds, basestring) else _dshape_builder(ds)
    (shape, dtype) = to_numpy(ds)
    datadesc = NumPyDataDescriptor(ones(shape, dtype=dtype))
    return Array(datadesc)


# for a temptative open function:
def open(uri):
    raise NotImplementedError
