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
from .datashape import dshape as _dshape_builder, to_numpy, to_dtype

import numpy as np
from . import blz
from .py3help import basestring, urlparse

try:
    basestring
    # if basestring exists... use it (fails on python 3)
    def _is_str(s):
        return isinstance(s, basestring)
except NameError:
    # python 3 version
    def _is_str(s):
        return isinstance(s, str)

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
    dshape = dshape if not _is_str(dshape) else _dshape_builder(dshape)

    if isinstance(obj, IDataDescriptor):
        # TODO: Validate the 'caps', convert to another kind
        #       of data descriptor if necessary
        # Note by Francesc: but if it is already an IDataDescriptor I wonder
        # if `caps` should be ignored.  Hmm, probably not...
        dd = obj
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
    elif isinstance(obj, np.ndarray):
        dd = NumPyDataDescriptor(obj)
    elif isinstance(obj, blz.barray):
        dd = BLZDataDescriptor(obj)
    else:
        raise TypeError(('Failed to construct blaze array from '
                        'object of type %r') % type(obj))
    return Array(dd)


# XXX This should probably be made public because the `count` param
# for BLZ is very important for getting good performance.
def _fromiter(gen, dshape, caps):
    """Create an array out of an iterator."""
    dshape = dshape if not _is_str(dshape) else _dshape_builder(dshape)

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
    dshape = dshape if not _is_str(dshape) else _dshape_builder(dshape)
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
    dshape = dshape if not _is_str(dshape) else _dshape_builder(dshape)

    if 'efficient-write' in caps:
        dd = NumPyDataDescriptor(np.ones(*to_numpy(dshape)))
    elif 'compress' in caps:
        shape, dt = to_numpy(dshape)
        dd = BLZDataDescriptor(blz.ones(*to_numpy(dshape)))
    return Array(dd)

# XXX A big hack for some quirks in current datashape. The next deals
# with the cases where the shape is not present like in 'float32'
def _to_numpy(ds):
    res = to_numpy(ds)
    res = res if type(res) is tuple else ((), to_dtype(ds))
    return res

# Persistent constructors:
def create(uri, dshape, caps={'efficient-append': True}):
    """Create a 0-length persistent array.

    Parameters
    ----------
    uri : URI string
        The URI of where the array will be stored (e.g. blz://myfile.blz).

    dshape : datashape
        The datashape for the resulting array.

    caps : capabilities dictionary
        A dictionary containing the desired capabilities of the array.

    Returns
    -------
    out: a concrete blaze array.

    Notes
    -----

    The shape part of the `dshape` is ignored.  This should be fixed
    by testing that the shape is actually empty.

    Only the BLZ format is supported currently.

    """
    dshape = dshape if not _is_str(dshape) else _dshape_builder(dshape)
    # Only BLZ supports efficient appends right now
    shape, dt = _to_numpy(dshape)
    shape = (0,) + shape  # the leading dimension will be 0
    uri = urlparse.urlparse(uri)
    path = uri.netloc + uri.path
    if 'efficient-append' in caps:
        dd = BLZDataDescriptor(blz.zeros(shape, dtype=dt, rootdir=path))
    elif 'efficient-write' in caps:
        raise ValueError('efficient-write objects not supported for '
                         'persistence')
    else:
        # BLZ will be the default
        dd = BLZDataDescriptor(blz.zeros(shape, dtype=dt, rootdir=path))
    return Array(dd)


def open(uri):
    """Open an existing persistent array.

    Parameters
    ----------
    uri : URI string
        The URI of where the array is stored (e.g. blz://myfile.blz).

    Returns
    -------
    out: a concrete blaze array.

    Notes
    -----
    Only the BLZ format is supported currently.

    """
    uri = urlparse.urlparse(uri)
    path = uri.netloc + uri.path
    d = blz.open(rootdir=path)
    dd = BLZDataDescriptor(d)
    return Array(dd)
