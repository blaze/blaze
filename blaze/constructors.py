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

import inspect

from .array import Array
from .datadescriptor import (IDataDescriptor,
                NumPyDataDescriptor, BLZDataDescriptor)
from .datashape import to_numpy, to_dtype

import numpy as np
from . import blz
from ._api_helpers import _normalize_dshape

# note that this is rather naive. In fact, a proper way to implement
# the array from a numpy is creating a ByteProvider based on "obj"
# and infer the indexer from the apropriate information in the numpy
# array.
def array(obj, dshape=None, caps={'efficient-write': True}):
    """Create an in-memory Blaze array.

    Parameters
    ----------
    obj : array_like
        Initial contents for the array.

    dshape : datashape
        The datashape for the resulting array. By default the
        datashape will be inferred from data. If an explicit dshape is
        provided, the input data will be coerced into the provided
        dshape.

        caps : capabilities dictionary
        A dictionary containing the desired capabilities of the array.

    Returns
    -------
    out : a concrete, in-memory blaze array.

    Bugs
    ----
    Right now the explicit dshape is ignored. This needs to be
    corrected. When the data cannot be coerced to an explicit dshape
    an exception should be raised.

    """
    dshape = _normalize_dshape(dshape)

    if isinstance(obj, IDataDescriptor):
        # TODO: Validate the 'caps', convert to another kind
        #       of data descriptor if necessary
        dd = obj
    elif isinstance(obj, np.ndarray):
        dd = NumPyDataDescriptor(obj)
    elif isinstance(obj, blz.barray):
        dd = BLZDataDescriptor(obj)
    elif inspect.isgenerator(obj):
        return _fromiter(obj, dshape, caps)
    elif 'efficient-write' in caps:
        dt = None if dshape is None else to_dtype(dshape)
        # NumPy provides efficient writes
        dd = NumPyDataDescriptor(np.array(obj, dtype=dt))
    elif 'compress' in caps:
        dt = None if dshape is None else to_dtype(dshape)
        # BLZ provides compression
        dd = BLZDataDescriptor(blz.barray(obj, dtype=dt))
    else:
        raise TypeError(('Failed to construct blaze array from '
                        'object of type %r') % type(obj))
    return Array(dd)


# XXX This should probably be made public because the `count` param
# for BLZ is very important for getting good performance.
def _fromiter(gen, dshape, caps):
    """Create an array out of an iterator."""
    dshape = _normalize_dshape(dshape)

    if 'efficient-write' in caps:
        dt = None if dshape is None else to_dtype(dshape)
        dd = NumPyDataDescriptor(np.fromiter(gen, dtype=dt))
    elif 'compress' in caps:
        dt = None if dshape is None else to_dtype(dshape)
        dd = BLZDataDescriptor(blz.fromiter(gen, dtype=dt, count=-1))
    return Array(dd)


def zeros(dshape, caps={'efficient-write': True}):
    """Create an array and fill it with zeros.

    Parameters
    ----------
    dshape : datashape
        The datashape for the resulting array.

    caps : capabilities dictionary
        A dictionary containing the desired capabilities of the array.

    Returns
    -------
    out : a concrete, in-memory blaze array.

    """
    dshape = _normalize_dshape(dshape)

    if 'efficient-write' in caps:
        dd = NumPyDataDescriptor(np.zeros(*to_numpy(dshape)))
    elif 'compress' in caps:
        dd = BLZDataDescriptor(blz.zeros(*to_numpy(dshape)))
    return Array(dd)


def ones(dshape, caps={'efficient-write': True}):
    """Create an array and fill it with ones.

    Parameters
    ----------
    dshape : datashape
        The datashape for the resulting array.

    caps : capabilities dictionary
        A dictionary containing the desired capabilities of the array.

    Returns
    -------
    out: a concrete blaze array.

    """
    dshape = _normalize_dshape(dshape)

    if 'efficient-write' in caps:
        dd = NumPyDataDescriptor(np.ones(*to_numpy(dshape)))
    elif 'compress' in caps:
        shape, dt = to_numpy(dshape)
        dd = BLZDataDescriptor(blz.ones(*to_numpy(dshape)))
    return Array(dd)

