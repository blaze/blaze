from __future__ import absolute_import, division, print_function

import ctypes

from dynd import ndt, _lowlevel
import datashape

from .dynd_data_descriptor import DyND_DDesc


def data_descriptor_from_ctypes(cdata, writable):
    """
    Parameters
    ----------
    cdata : ctypes data instance
        The ctypes data object which owns the data.
    writable : bool
        Should be true if the data is writable, flase
        if it's read-only.
    """
    ds = datashape.from_ctypes(type(cdata))
    access = "readwrite" if writable else "readonly"
    dyndarr = _lowlevel.array_from_ptr(ndt.type(str(ds)),
                    ctypes.addressof(cdata), cdata,
                    access)
    return DyND_DDesc(dyndarr)


def data_descriptor_from_cffi(ffi, cdata, writable):
    """
    Parameters
    ----------
    ffi : cffi.FFI
        The cffi namespace which contains the cdata.
    cdata : cffi.CData
        The cffi data object which owns the data.
    writable : bool
        Should be true if the data is writable, flase
        if it's read-only.
    """
    if not isinstance(cdata, ffi.CData):
        raise TypeError('object is not a cffi.CData object, has type %s' %
                        type(cdata))
    owner = (ffi, cdata)
    # Get the raw pointer out of the cdata as an integer
    ptr = int(ffi.cast('uintptr_t', ffi.cast('char *', cdata)))
    ds = datashape.from_cffi(ffi, ffi.typeof(cdata))
    if (isinstance(ds, datashape.DataShape) and
            isinstance(ds[0], datashape.TypeVar)):
        # If the outermost dimension is an array without fixed
        # size, get its size from the data
        ds = datashape.DataShape(*(datashape.Fixed(len(cdata)),) + ds[1:])
    access = "readwrite" if writable else "readonly"
    dyndarr = _lowlevel.array_from_ptr(ndt.type(str(ds)), ptr, owner, access)
    return DyND_DDesc(dyndarr)

