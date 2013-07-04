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
                DyNDDataDescriptor, BLZDataDescriptor)
from .datashape import to_numpy, to_numpy_dtype
from .storage import Storage

from dynd import nd, ndt
import numpy as np
from . import blz
from ._api_helpers import _normalize_dshape

# note that this is rather naive. In fact, a proper way to implement
# the array from a numpy is creating a ByteProvider based on "obj"
# and infer the indexer from the apropriate information in the numpy
# array.
def array(obj, dshape=None, caps={'efficient-write': True},
          storage=None):
    """Create a Blaze array.

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

    storage : Storage instance
        A Storage object with the necessary info for storing the data.

    Returns
    -------
    out : a concrete blaze array.

    Bugs
    ----
    Right now the explicit dshape is ignored. This needs to be
    corrected. When the data cannot be coerced to an explicit dshape
    an exception should be raised.

    """
    dshape = _normalize_dshape(dshape)

    storage = _storage_convert(storage)

    if isinstance(obj, IDataDescriptor):
        # TODO: Validate the 'caps', convert to another kind
        #       of data descriptor if necessary
        # Note by Francesc: but if it is already an IDataDescriptor I wonder
        # if `caps` should be ignored.  Hmm, probably not...
        #
        # Note by Oscar: Maybe we shouldn't accept a datadescriptor at
        #   all at this level. If you've got a DataDescriptor you are
        #   playing with internal datastructures anyways, go to the
        #   Array constructor directly. If you want to transform to
        #   another datadescriptor... convert it yourself (you are
        #   playing with internal datastructures, remember? you should
        #   be able to do it in your own.
        dd = obj
    elif inspect.isgenerator(obj):
        return _fromiter(obj, dshape, caps, storage)
    elif storage is not None:
        dt = None if dshape is None else to_numpy_dtype(dshape)
        dd = BLZDataDescriptor(
            blz.barray(obj, dtype=dt, rootdir=storage.path))
    elif 'efficient-write' in caps and caps['efficient-write'] is True:
        # In-Memory array
        if dshape is None:
            dd = DyNDDataDescriptor(nd.array(obj))
        else:
            dd = DyNDDataDescriptor(nd.array(obj, udtype=str(dshape)))
    elif 'compress' in caps and caps['compress'] is True:
        dt = None if dshape is None else to_numpy_dtype(dshape)
        # BLZ provides compression
        dd = BLZDataDescriptor(blz.barray(obj, dtype=dt))
    elif isinstance(obj, np.ndarray):
        dd = DyNDDataDescriptor(nd.array(obj))
    elif isinstance(obj, nd.array):
        dd = DyNDDataDescriptor(obj)
    elif isinstance(obj, blz.barray):
        dd = BLZDataDescriptor(obj)
    else:
        raise TypeError(('Failed to construct blaze array from '
                        'object of type %r') % type(obj))
    return Array(dd)


def _storage_convert(storage):
    if storage is not None and isinstance(storage, str):
        storage = Storage(storage)
    return storage


# XXX This should probably be made public because the `count` param
# for BLZ is very important for getting good performance.
def _fromiter(gen, dshape, caps, storage):
    """Create an array out of an iterator."""
    dshape = _normalize_dshape(dshape)

    # TODO: deal with non-supported capabilities.  Perhaps it would be
    # better to convert caps into a class to check for supported
    # capabilities only.
    if storage is not None:
        dt = None if dshape is None else to_numpy_dtype(dshape)
        dd = BLZDataDescriptor(blz.barray(gen, dtype=dt, count=-1,
                                          rootdir=storage.path))
    elif 'efficient-write' in caps and caps['efficient-write'] is True:
        dt = None if dshape is None else to_numpy_dtype(dshape)
        # TODO: add a var dimension ahead of the dshape and
        #       construct as dynd.
        dd = DyNDDataDescriptor(nd.array(np.fromiter(gen, dtype=dt)))
    elif 'compress' in caps and caps['compress'] is True:
        dt = None if dshape is None else to_numpy_dtype(dshape)
        dd = BLZDataDescriptor(blz.fromiter(gen, dtype=dt, count=-1))
    else:
        # Fall-back is NumPy
        dt = None if dshape is None else to_numpy_dtype(dshape)
        # TODO: add a var dimension ahead of the dshape and
        #       construct as dynd.
        dd = DyNDDataDescriptor(nd.array(np.fromiter(gen, dtype=dt)))

    return Array(dd)


def empty(dshape, caps={'efficient-write': True}, storage=None):
    """Create an array with uninitialized data.

    Parameters
    ----------
    dshape : datashape
        The datashape for the resulting array.

    caps : capabilities dictionary
        A dictionary containing the desired capabilities of the array.

    storage : Storage instance
        A Storage object with the necessary info for data storage.

    Returns
    -------
    out : a concrete blaze array.

    """
    dshape = _normalize_dshape(dshape)
    storage = _storage_convert(storage)

    if storage is not None:
        shape, dt = to_numpy(dshape)
        dd = BLZDataDescriptor(blz.zeros(shape, dt,
                                         rootdir=storage.path))
    elif 'efficient-write' in caps:
        dd = DyNDDataDescriptor(nd.empty(str(dshape)))
    elif 'compress' in caps:
        dd = BLZDataDescriptor(blz.zeros(shape, dt))
    return Array(dd)

def zeros(dshape, caps={'efficient-write': True}, storage=None):
    """Create an array and fill it with zeros.

    Parameters
    ----------
    dshape : datashape
        The datashape for the resulting array.

    caps : capabilities dictionary
        A dictionary containing the desired capabilities of the array.

    storage : Storage instance
        A Storage object with the necessary info for data storage.

    Returns
    -------
    out : a concrete blaze array.

    """
    dshape = _normalize_dshape(dshape)
    storage = _storage_convert(storage)

    if storage is not None:
        shape, dt = to_numpy(dshape)
        dd = BLZDataDescriptor(blz.zeros(shape, dt,
                                         rootdir=storage.path))
    elif 'efficient-write' in caps:
        # TODO: Handle var dimension properly (raise exception?)
        dyndarr = nd.empty(str(dshape))
        dyndarr[...] = False
        dd = DyNDDataDescriptor(dyndarr)
    elif 'compress' in caps:
        shape, dt = to_numpy(dshape)
        dd = BLZDataDescriptor(blz.zeros(shape, dt))
    return Array(dd)


def ones(dshape, caps={'efficient-write': True}, storage=None):
    """Create an array and fill it with ones.

    Parameters
    ----------
    dshape : datashape
        The datashape for the resulting array.

    caps : capabilities dictionary
        A dictionary containing the desired capabilities of the array.

    storage : Storage instance
        A Storage object with the necessary info for data storage.

    Returns
    -------
    out: a concrete blaze array.

    """
    dshape = _normalize_dshape(dshape)
    storage = _storage_convert(storage)

    if storage is not None:
        shape, dt = to_numpy(dshape)
        dd = BLZDataDescriptor(blz.ones(shape, dt,
                                        rootdir=storage.path))
    elif 'efficient-write' in caps:
        # TODO: Handle var dimension properly (raise exception?)
        dyndarr = nd.empty(str(dshape))
        dyndarr[...] = True
        dd = DyNDDataDescriptor(dyndarr)
    elif 'compress' in caps:
        shape, dt = to_numpy(dshape)
        dd = BLZDataDescriptor(blz.ones(shape, dt))
    return Array(dd)
