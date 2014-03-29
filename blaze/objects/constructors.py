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
    IDataDescriptor, DyNDDataDescriptor, BLZDataDescriptor, HDF5DataDescriptor)
from ..io.storage import Storage
from ..py2help import basestring


def split_path(dp):
    """Split a path in rootdir path and end part for HDF5 purposes"""
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


def array(obj, dshape=None, dd=None):
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

    dd : data descriptor instance
        This comes with the necessary info for storing the data.  If
        None, a DyNDDataDescriptor will be used.

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

    if isinstance(obj, Array):
        return obj
    elif isinstance(obj, IDataDescriptor):
        if dd is None:
            dd = obj
            return Array(dd)
        else:
            raise ValueError(('you cannot specify `dd` when `obj` '
                              'is already a DataDescriptor'))

    if dd is None:
        # Use a dynd dd by default
        try:
            if dshape is None:
                array = nd.asarray(obj, access='rw')
            else:
                # Use the uniform/full dtype specification in dynd depending
                # on whether the datashape has a uniform dim
                dt = ndt.type(str(dshape))
                if dt.ndim > 0:
                    array = nd.array(obj, type=dt, access='rw')
                else:
                    array = nd.array(obj, dtype=dt, access='rw')
        except:
            raise ValueError(('failed to construct a dynd array from '
                              'object %r') % obj)
        dd = DyNDDataDescriptor(array)
        return Array(dd)

    dt = None if dshape is None else to_numpy_dtype(dshape)
    if isinstance(dd, BLZDataDescriptor):
        if inspect.isgenerator(obj):
            # TODO: Generator logic could go inside barray
            dd.blzarr = blz.fromiter(obj, dtype=dt, count=-1,
                                    rootdir=dd.path, mode=dd.mode,
                                    **dd.kwargs)
        else:
            dd.blzarr = blz.barray(
                obj, dtype=dt, rootdir=dd.path, mode=dd.mode, **dd.kwargs)
    elif isinstance(dd, HDF5DataDescriptor):
        with tb.open_file(dd.path, mode=dd.mode) as f:
            where, name = split_path(dd.datapath)
            f.create_earray(where, name, filters=dd.filters, obj=obj)

    return Array(dd)


# TODO: Make overloaded constructors, taking dshape, **kwds. Overload
# on keywords

def empty(dshape, dd=None):
    """Create an array with uninitialized data.

    Parameters
    ----------
    dshape : datashape
        The datashape for the resulting array.

    dd : data descriptor instance
        This comes with the necessary info for storing the data.  If
        None, a DyNDDataDescriptor will be used.

    Returns
    -------
    out : a concrete blaze array.

    """
    dshape = _normalize_dshape(dshape)

    if dd is None:
        dd = DyNDDataDescriptor(nd.empty(str(dshape)))
        return Array(dd)
    if isinstance(dd, BLZDataDescriptor):
        shape, dt = to_numpy(dshape)
        dd.blzarr = blz.zeros(shape, dt, rootdir=dd.path,
                              mode=dd.mode, **dd.kwargs)
    elif isinstance(dd, HDF5DataDescriptor):
        obj = nd.as_numpy(nd.empty(str(dshape)))
        with tb.open_file(dd.path, mode=dd.mode) as f:
            where, name = split_path(dd.datapath)
            f.create_earray(where, name, filters=dd.filters, obj=obj)
    return Array(dd)


def zeros(dshape, dd=None):
    """Create an array and fill it with zeros.

    Parameters
    ----------
    dshape : datashape
        The datashape for the resulting array.

    dd : data descriptor instance
        This comes with the necessary info for storing the data.  If
        None, a DyNDDataDescriptor will be used.

    Returns
    -------
    out : a concrete blaze array.

    """
    dshape = _normalize_dshape(dshape)

    if dd is None:
        dd = DyNDDataDescriptor(nd.zeros(str(dshape)))
        return Array(dd)
    if isinstance(dd, BLZDataDescriptor):
        shape, dt = to_numpy(dshape)
        dd.blzarr = blz.zeros(shape, dt, rootdir=dd.path, mode=dd.mode,
                              **dd.kwargs)
    elif isinstance(dd, HDF5DataDescriptor):
        obj = nd.as_numpy(nd.zeros(str(dshape)))
        with tb.open_file(dd.path, mode=dd.mode) as f:
            where, name = split_path(dd.datapath)
            f.create_earray(where, name, filters=dd.filters, obj=obj)
    return Array(dd)


def ones(dshape, dd=None):
    """Create an array and fill it with ones.

    Parameters
    ----------
    dshape : datashape
        The datashape for the resulting array.

    dd : data descriptor instance
        This comes with the necessary info for storing the data.  If
        None, a DyNDDataDescriptor will be used.

    Returns
    -------
    out: a concrete blaze array.

    """
    dshape = _normalize_dshape(dshape)

    if dd is None:
        dd = DyNDDataDescriptor(nd.ones(str(dshape)))
        return Array(dd)
    if isinstance(dd, BLZDataDescriptor):
        shape, dt = to_numpy(dshape)
        dd.blzarr = blz.ones(shape, dt, rootdir=dd.path, mode=dd.mode,
                             **dd.kwargs)
    elif isinstance(dd, HDF5DataDescriptor):
        obj = nd.as_numpy(nd.empty(str(dshape)))
        with tb.open_file(dd.path, mode=dd.mode) as f:
            where, name = split_path(dd.datapath)
            f.create_earray(where, name, filters=dd.filters, obj=obj)
    return Array(dd)


def drop(dd):
    """Remove a persistent storage.

    Parameters
    ----------
    dd : data descriptor instance
        This comes with the necessary info for opening the data stored.

    """

    if isinstance(dd, BLZDataDescriptor):
        from shutil import rmtree
        rmtree(dd.path)
    else:
        import os
        os.unlink(dd.path)
