"""Constructors for the blaze array object.

Having them as external functions allows to more flexibility and helps keeping
the blaze array object compact, just showing the interface of the
array itself.

The blaze array __init__ method should be considered private and for
advanced users only. It will provide the tools supporting the rest
of the constructors, and will use low-level parameters, like
ByteProviders, that an end user may not even need to know about.
"""

from __future__ import absolute_import, division, print_function

import inspect

from dynd import nd, ndt
import numpy as np
import datashape
from datashape import to_numpy, to_numpy_dtype
import blz

from ..optional_packages import tables_is_here
if tables_is_here:
    import tables as tb

from .array import Array
from ..datadescriptor import (
    DDesc, DyND_DDesc, BLZ_DDesc, HDF5_DDesc)
from ..py2help import basestring


def split_path(dp):
    """Split a path in basedir path and end part for HDF5 purposes"""
    idx = dp.rfind('/')
    where = dp[:idx] if idx > 0 else '/'
    name = dp[idx+1:]
    return where, name


def _normalize_dshape(ds):
    """
    In the API, when a datashape is provided we want to support
    them in string form as well. This function will convert from any
    form we want to support in the API inputs into the internal
    datashape object, so the logic is centralized in a single
    place. Any API function that receives a dshape as a parameter
    should convert it using this function.
    """
    if isinstance(ds, basestring):
        return datashape.dshape(ds)
    else:
        return ds


def array(obj, dshape=None, ddesc=None):
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

    ddesc : data descriptor instance
        This comes with the necessary info for storing the data.  If
        None, a DyND_DDesc will be used.

    Returns
    -------
    out : a concrete blaze array.

    """
    dshape = _normalize_dshape(dshape)

    if ((obj is not None) and
        (not inspect.isgenerator(obj)) and
        (dshape is not None)):
        dt = ndt.type(str(dshape))
        if dt.ndim > 0:
            obj = nd.array(obj, type=dt, access='rw')
        else:
            obj = nd.array(obj, dtype=dt, access='rw')

    if obj is None and ddesc is None:
        raise ValueError('you need to specify at least `obj` or `ddesc`')

    if isinstance(obj, Array):
        return obj
    elif isinstance(obj, DDesc):
        if ddesc is None:
            ddesc = obj
            return Array(ddesc)
        else:
            raise ValueError(('you cannot specify `ddesc` when `obj` '
                              'is already a DDesc instance'))

    if ddesc is None:
        # Use a dynd ddesc by default
        try:
            array = nd.asarray(obj, access='rw')
        except:
            raise ValueError(('failed to construct a dynd array from '
                              'object %r') % obj)
        ddesc = DyND_DDesc(array)
        return Array(ddesc)

    # The DDesc has been specified
    if isinstance(ddesc, DyND_DDesc):
        if obj is not None:
            raise ValueError(('you cannot specify simultaneously '
                              '`obj` and a DyND `ddesc`'))
        return Array(ddesc)
    elif isinstance(ddesc, BLZ_DDesc):
        if inspect.isgenerator(obj):
            dt = None if dshape is None else to_numpy_dtype(dshape)
            # TODO: Generator logic could go inside barray
            ddesc.blzarr = blz.fromiter(obj, dtype=dt, count=-1,
                                        rootdir=ddesc.path, mode=ddesc.mode,
                                        **ddesc.kwargs)
        else:
            if isinstance(obj, nd.array):
                obj = nd.as_numpy(obj)
            ddesc.blzarr = blz.barray(
                obj, rootdir=ddesc.path, mode=ddesc.mode, **ddesc.kwargs)
    elif isinstance(ddesc, HDF5_DDesc):
        if isinstance(obj, nd.array):
            obj = nd.as_numpy(obj)
        with tb.open_file(ddesc.path, mode=ddesc.mode) as f:
            where, name = split_path(ddesc.datapath)
            f.create_earray(where, name, filters=ddesc.filters, obj=obj)

    return Array(ddesc)


# TODO: Make overloaded constructors, taking dshape, **kwds. Overload
# on keywords

def empty(dshape, ddesc=None):
    """Create an array with uninitialized data.

    Parameters
    ----------
    dshape : datashape
        The datashape for the resulting array.

    ddesc : data descriptor instance
        This comes with the necessary info for storing the data.  If
        None, a DyND_DDesc will be used.

    Returns
    -------
    out : a concrete blaze array.

    """
    dshape = _normalize_dshape(dshape)

    if ddesc is None:
        ddesc = DyND_DDesc(nd.empty(str(dshape)))
        return Array(ddesc)
    if isinstance(ddesc, BLZ_DDesc):
        shape, dt = to_numpy(dshape)
        ddesc.blzarr = blz.zeros(shape, dt, rootdir=ddesc.path,
                                 mode=ddesc.mode, **ddesc.kwargs)
    elif isinstance(ddesc, HDF5_DDesc):
        obj = nd.as_numpy(nd.empty(str(dshape)))
        with tb.open_file(ddesc.path, mode=ddesc.mode) as f:
            where, name = split_path(ddesc.datapath)
            f.create_earray(where, name, filters=ddesc.filters, obj=obj)
    return Array(ddesc)


def zeros(dshape, ddesc=None):
    """Create an array and fill it with zeros.

    Parameters
    ----------
    dshape : datashape
        The datashape for the resulting array.

    ddesc : data descriptor instance
        This comes with the necessary info for storing the data.  If
        None, a DyND_DDesc will be used.

    Returns
    -------
    out : a concrete blaze array.

    """
    dshape = _normalize_dshape(dshape)

    if ddesc is None:
        ddesc = DyND_DDesc(nd.zeros(str(dshape), access='rw'))
        return Array(ddesc)
    if isinstance(ddesc, BLZ_DDesc):
        shape, dt = to_numpy(dshape)
        ddesc.blzarr = blz.zeros(
            shape, dt, rootdir=ddesc.path, mode=ddesc.mode, **ddesc.kwargs)
    elif isinstance(ddesc, HDF5_DDesc):
        obj = nd.as_numpy(nd.zeros(str(dshape)))
        with tb.open_file(ddesc.path, mode=ddesc.mode) as f:
            where, name = split_path(ddesc.datapath)
            f.create_earray(where, name, filters=ddesc.filters, obj=obj)
    return Array(ddesc)


def ones(dshape, ddesc=None):
    """Create an array and fill it with ones.

    Parameters
    ----------
    dshape : datashape
        The datashape for the resulting array.

    ddesc : data descriptor instance
        This comes with the necessary info for storing the data.  If
        None, a DyND_DDesc will be used.

    Returns
    -------
    out: a concrete blaze array.

    """
    dshape = _normalize_dshape(dshape)

    if ddesc is None:
        ddesc = DyND_DDesc(nd.ones(str(dshape), access='rw'))
        return Array(ddesc)
    if isinstance(ddesc, BLZ_DDesc):
        shape, dt = to_numpy(dshape)
        ddesc.blzarr = blz.ones(
            shape, dt, rootdir=ddesc.path, mode=ddesc.mode, **ddesc.kwargs)
    elif isinstance(ddesc, HDF5_DDesc):
        obj = nd.as_numpy(nd.empty(str(dshape)))
        with tb.open_file(ddesc.path, mode=ddesc.mode) as f:
            where, name = split_path(ddesc.datapath)
            f.create_earray(where, name, filters=ddesc.filters, obj=obj)
    return Array(ddesc)


def drop(obj):
    """Remove a persistent storage based on datadescriptor info.

    Parameters
    ----------
    obj : Array or data descriptor instance
        The Array or data descriptor to be removed.

    """

    if isinstance(obj, Array):
        ddesc = obj.ddesc
    elif isinstance(obj, DDesc):
        ddesc = obj
    else:
        raise ValueError("`obj` must be an Array or DDesc instance")

    if isinstance(ddesc, BLZ_DDesc):
        from shutil import rmtree
        rmtree(ddesc.path)
    else:
        import os
        os.unlink(ddesc.path)
