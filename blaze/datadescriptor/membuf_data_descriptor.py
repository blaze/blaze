from __future__ import absolute_import
import operator

import sys
from blaze import dshape, datashape
from . import (IElementReader, IElementWriter,
                IElementReadIter, IElementWriteIter,
                IDataDescriptor)
from ..datashape import DataShape
import ctypes

if sys.version_info >= (3, 0):
    _inttypes = (int,)
else:
    _inttypes = (int, long)

def membuf_descriptor_iter(mbdd):
    c_outer_stride = mbdd.dshape.c_strides[0]
    end = mbdd.ptr + operator.index(mbdd.dshape[0]) * c_outer_stride
    # Create the data shape of the result
    ds = mbdd.dshape
    if len(ds) == 2:
        ds = ds[-1]
    elif len(ds) > 2:
        ds = datashape.DataShape(ds[1:])
    else:
        raise IndexError('Cannot iterator over a scalar')

    for ptr in range(mbdd.ptr, end, c_outer_stride):
        yield MemBufDataDescriptor(ptr, mbdd.ptr_owner, ds)

class MemBufElementReader(IElementReader):
    def __init__(self, mbdd, nindex):
        if nindex > len(mbdd.dshape) - 1:
            raise IndexError('Cannot have more indices than dimensions')
        self._mbdd = mbdd
        if nindex == len(mbdd.dshape) - 1:
            self._dshape = mbdd.dshape[-1]
        else:
            self._dshape = DataShape(mbdd.dshape[nindex:])
        self._nindex = nindex

    @property
    def dshape(self):
        return self._dshape

    @property
    def nindex(self):
        return self._nindex

    def read_single(self, idx):
        if len(idx) != self.nindex:
            raise IndexError(('Incorrect number of indices '
                            '(got %d, require %d)') % (len(idx), self.nindex))
        if self.nindex > 0:
            idx = tuple([operator.index(i) for i in idx])
            c_strides = self._mbdd.dshape.c_strides[:self.nindex]
            offset = sum(stride * idx for stride, idx in zip(c_strides, idx))
            return self._mbdd.ptr + offset
        else:
            return self._mbdd.ptr

class MemBufElementReadIter(IElementReadIter):
    def __init__(self, mbdd):
        if len(mbdd.dshape) <= 1:
            raise IndexError('Need at least one dimension for iteration')
        self._outer_stride = mbdd.dshape.c_strides[0]
        self._mbdd = mbdd
        if len(mbdd.dshape) == 2:
            self._dshape = mbdd.dshape[-1]
        else:
            self._dshape = DataShape(mbdd.dshape[1:])
        self._ptr = mbdd.ptr
        self._end = (self._ptr +
                        self._outer_stride * operator.index(mbdd.dshape[0]))

    @property
    def dshape(self):
        return self._dshape

    def __len__(self):
        return self._len

    def __next__(self):
        if self._ptr < self._end:
            result = self._ptr
            self._ptr += self._outer_stride
            return result
        else:
            raise StopIteration

class MemBufDataDescriptor(IDataDescriptor):
    """
    A Blaze data descriptor which exposes a raw memory buffer,
    which is in C-order with C struct alignment.
    """
    def __init__(self, ptr, ptr_owner, ds):
        """
        Parameters
        ----------
        ptr : int/long
            A raw pointer to the data. This data must be
            in C-order and follow C struct alignment,
            matching a C interpretation of the datashape.
        ds : Blaze dshape
            The data shape of the data pointed to by ptr.
        """
        assert isinstance(ptr, _inttypes)
        self._ptr = ptr
        self._ptr_owner = ptr_owner
        self._dshape = ds

    @property
    def dshape(self):
        return self._dshape

    @property
    def ptr(self):
        return self._ptr

    @property
    def ptr_owner(self):
        return self._ptr_owner

    def __len__(self):
        if isinstance(self._dshape, datashape.DataShape):
            return operator.index(self._dshape[0])
        else:
            raise IndexError('Cannot get the length of a '
                            'zero-dimensional array')

    def __getitem__(self, key):
        # Just integer indices (no slices) for now
        if not isinstance(key, tuple):
            key = (key,)
        key = tuple([operator.index(i) for i in key])
        if len(key) > len(self._dshape) - 1:
           raise IndexError('Cannot have more indices than dimensions')

        # Apply each index in turn
        ds = self._dshape
        c_strides = ds.c_strides
        ptr = self._ptr
        for i, idx in enumerate(key):
            dim_size = operator.index(ds[i])
            # Implement Python's negative indexing
            if idx >= 0:
                if idx >= dim_size:
                    raise IndexError(('Index %d is out of range '
                                    'in dimension sized %d') % (idx, dim_size))
            else:
                if idx < -dim_size:
                    raise IndexError(('Index %d is out of range '
                                    'in dimension sized %d') % (idx, dim_size))
                idx += dim_size
            ptr = ptr + idx * c_strides[i]
        # Create the data shape of the result
        if len(key) == len(ds) - 1:
            ds = ds[-1]
        else:
            ds = datashape.DataShape(ds[len(key):])
        return MemBufDataDescriptor(ptr, self.ptr_owner, ds)

    def __iter__(self):
        return membuf_descriptor_iter(self)

    def element_reader(self, nindex):
        return MemBufElementReader(self, nindex)

    def element_read_iter(self):
        return MemBufElementReadIter(self)

def data_descriptor_from_ctypes(cdata):
    """
    Parameters
    ----------
    cdata : ctypes data instance
        The ctypes data object which owns the data.
    """
    return MemBufDataDescriptor(ctypes.addressof(cdata), cdata,
                    datashape.from_ctypes(type(cdata)))

def data_descriptor_from_cffi(ffi, cdata):
    """
    Parameters
    ----------
    ffi : cffi.FFI
        The cffi namespace which contains the cdata.
    cdata : cffi.CData
        The cffi data object which owns the data.
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
        ds = datashape.DataShape((datashape.Fixed(len(cdata)),) + ds[1:])
    return MemBufDataDescriptor(ptr, owner, ds)

