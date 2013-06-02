from __future__ import absolute_import
import operator
import contextlib
import sys
import ctypes

from blaze import dshape, datashape
from .data_descriptor import (IElementReader, IElementWriter,
                IElementReadIter, IElementWriteIter,
                IDataDescriptor, buffered_ptr_ctxmgr)
from ..datashape import DataShape

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
        raise IndexError('Cannot iterate over a scalar')

    for ptr in range(mbdd.ptr, end, c_outer_stride):
        yield MemBufDataDescriptor(ptr, mbdd.ptr_owner, ds, mbdd._writable)

class MemBufElementReader(IElementReader):
    def __init__(self, mbdd, nindex):
        if nindex > len(mbdd.dshape) - 1:
            raise IndexError('Cannot have more indices than dimensions')
        self._mbdd = mbdd
        self._dshape = mbdd.dshape.subarray(nindex)
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

class MemBufElementWriter(IElementWriter):
    def __init__(self, mbdd, nindex):
        if nindex > len(mbdd.dshape) - 1:
            raise IndexError('Cannot have more indices than dimensions')
        self._mbdd = mbdd
        self._dshape = mbdd.dshape.subarray(nindex)
        self._nindex = nindex

    @property
    def dshape(self):
        return self._dshape

    @property
    def nindex(self):
        return self._nindex

    def _get_item_ptr(self, idx):
        if len(idx) != self.nindex:
            raise IndexError('Incorrect number of indices (got %d, require %d)' %
                           (len(idx), self.nindex))
        idx = tuple(operator.index(i) for i in idx)
        if self.nindex > 0:
            idx = tuple([operator.index(i) for i in idx])
            c_strides = self._mbdd.dshape.c_strides[:self.nindex]
            offset = sum(stride * idx for stride, idx in zip(c_strides, idx))
            return self._mbdd.ptr + offset
        else:
            return self._mbdd.ptr

    def write_single(self, idx, ptr):
        # The memory is all in C order, so just a memcopy is needed
        ctypes.memmove(self._get_item_ptr(idx), ptr, self._dshape.c_itemsize)

    def buffered_ptr(self, idx):
        # The membuf is always in C format, so no
        # buffering is ever needed
        return buffered_ptr_ctxmgr(self._get_item_ptr(idx), None)

class MemBufElementReadIter(IElementReadIter):
    def __init__(self, mbdd):
        if len(mbdd.dshape) <= 1:
            raise IndexError('Need at least one dimension for iteration')
        self._outer_stride = mbdd.dshape.c_strides[0]
        self._mbdd = mbdd
        self._dshape = mbdd.dshape.subarray(1)
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

class MemBufElementWriteIter(IElementWriteIter):
    def __init__(self, mbdd):
        if len(mbdd.dshape) <= 1:
            raise IndexError('Need at least one dimension for iteration')
        self._outer_stride = mbdd.dshape.c_strides[0]
        self._mbdd = mbdd
        self._dshape = mbdd.dshape.subarray(1)
        self._ptr = mbdd.ptr
        self._end = (self._ptr +
                        self._outer_stride * operator.index(mbdd.dshape[0]))

    @property
    def dshape(self):
        return self._dshape

    def __len__(self):
        return self._len

    def __next__(self):
        # Because the membuf data is always in C
        # layout, it never needs buffering
        if self._ptr < self._end:
            result = self._ptr
            self._ptr += self._outer_stride
            return result
        else:
            raise StopIteration

    def close(self):
        pass

class MemBufDataDescriptor(IDataDescriptor):
    """
    A Blaze data descriptor which exposes a raw memory buffer,
    which is in C-order with C struct alignment.
    """
    def __init__(self, ptr, ptr_owner, ds, writable):
        """
        Parameters
        ----------
        ptr : int/long
            A raw pointer to the data. This data must be
            in C-order and follow C struct alignment,
            matching a C interpretation of the datashape.
        ptr_owner : object
            An object which owns or holds a reference to
            the data pointed to by `ptr`.
        ds : Blaze dshape
            The data shape of the data pointed to by ptr.
        writable : bool
            Should be true if the data is writable, flase
            if it's read-only.
        """
        assert isinstance(ptr, _inttypes)
        self._ptr = ptr
        self._ptr_owner = ptr_owner
        self._dshape = ds
        self._writable = bool(writable)

    @property
    def dshape(self):
        return self._dshape

    @property
    def writable(self):
        return self._writable

    @property
    def immutable(self):
        return False

    @property
    def ptr(self):
        return self._ptr

    @property
    def ptr_owner(self):
        return self._ptr_owner

    def __len__(self):
        if len(self._dshape) > 1:
            return operator.index(self._dshape[0])
        else:
            raise IndexError('Cannot get the length of a ' +
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
                    raise IndexError(('Index %d is out of range ' +
                                    'in dimension sized %d') % (idx, dim_size))
            else:
                if idx < -dim_size:
                    raise IndexError(('Index %d is out of range ' +
                                    'in dimension sized %d') % (idx, dim_size))
                idx += dim_size
            ptr = ptr + idx * c_strides[i]
        # Create the data shape of the result
        ds = ds.subarray(len(key))
        return MemBufDataDescriptor(ptr, self.ptr_owner, ds, self._writable)

    def __iter__(self):
        if len(self._dshape) > 1:
            return membuf_descriptor_iter(self)
        else:
            raise IndexError('Cannot iterate over a ' +
                            'zero-dimensional array')

    def element_reader(self, nindex):
        return MemBufElementReader(self, nindex)

    def element_read_iter(self):
        return MemBufElementReadIter(self)

    def element_writer(self, nindex):
        if self.writable:
            return MemBufElementWriter(self, nindex)
        else:
            raise ArrayWriteError('Cannot write to readonly MemBuf array')

    def element_write_iter(self):
        if self.writable:
            return contextlib.closing(MemBufElementWriteIter(self))
        else:
            raise ArrayWriteError('Cannot write to readonly MemBuf array')

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
    return MemBufDataDescriptor(ctypes.addressof(cdata), cdata,
                    datashape.from_ctypes(type(cdata)), writable)

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
        ds = datashape.DataShape((datashape.Fixed(len(cdata)),) + ds[1:])
    return MemBufDataDescriptor(ptr, owner, ds, writable)

