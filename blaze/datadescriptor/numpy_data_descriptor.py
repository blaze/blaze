from __future__ import absolute_import
import operator
import contextlib
import ctypes
import numpy as np

from .data_descriptor import (IElementReader, IElementWriter,
                IElementReadIter, IElementWriteIter,
                IDataDescriptor, buffered_ptr_ctxmgr)
from .. import datashape
from ..datashape import dshape
from ..error import ArrayWriteError

def numpy_descriptor_iter(npyarr):
    if npyarr.ndim > 1:
        for el in npyarr:
            yield NumPyDataDescriptor(el)
    else:
        for i in range(npyarr.shape[0]):
            # NumPy doesn't have a convenient way to avoid collapsing
            # to a scalar, this is a way to avoid that
            el = npyarr[...,np.newaxis][i].reshape(())
            yield NumPyDataDescriptor(el)

class NumPyElementReader(IElementReader):
    def __init__(self, npyarr, nindex):
        if nindex > npyarr.ndim:
            raise IndexError('Cannot have more indices than dimensions')
        self._nindex = nindex
        self._dshape = datashape.from_numpy(npyarr.shape[nindex:], npyarr.dtype)
        self.npyarr = npyarr

    @property
    def dshape(self):
        return self._dshape

    @property
    def nindex(self):
        return self._nindex

    def read_single(self, idx):
        if len(idx) != self.nindex:
            raise IndexError('Incorrect number of indices (got %d, require %d)' %
                           (len(idx), self.nindex))
        idx = tuple([operator.index(i) for i in idx])
        x = self.npyarr[idx]
        # Make it C-contiguous and in native byte order
        x = np.ascontiguousarray(x, dtype=x.dtype.newbyteorder('='))
        self._tmpbuffer = x
        return x.ctypes.data

class NumPyElementWriter(IElementWriter):
    def __init__(self, npyarr, nindex):
        if nindex > npyarr.ndim:
            raise IndexError('Cannot have more indices than dimensions')
        self._nindex = nindex
        self._shape = npyarr.shape[nindex:]
        self._dtype = npyarr.dtype.newbyteorder('=')
        self._element_size = np.prod(self._shape) * self._dtype.itemsize
        self._dshape = datashape.from_numpy(self._shape, self._dtype)
        self.npyarr = npyarr

    @property
    def dshape(self):
        return self._dshape

    @property
    def nindex(self):
        return self._nindex

    def write_single(self, idx, ptr):
        if len(idx) != self.nindex:
            raise IndexError('Incorrect number of indices (got %d, require %d)' %
                           (len(idx), self.nindex))
        idx = tuple(operator.index(i) for i in idx)
        # Create a temporary NumPy array around the ptr data
        buf = (ctypes.c_char * self._element_size).from_address(ptr)
        tmp = np.frombuffer(buf, self._dtype).reshape(self._shape)
        # Use NumPy's assignment to set the values
        self.npyarr[idx] = tmp

    def buffered_ptr(self, idx):
        if len(idx) != self.nindex:
            raise IndexError('Incorrect number of indices (got %d, require %d)' %
                           (len(idx), self.nindex))
        # Index into the numpy array without producing a scalar
        if len(idx) == self.npyarr.ndim:
            dst_arr = self.npyarr[...,np.newaxis][idx].reshape(())
        else:
            dst_arr = self.npyarr[idx]
        if dst_arr.flags.c_contiguous and dst_arr.dtype.isnative:
            # If no buffering is needed, just return the pointer
            return buffered_ptr_ctxmgr(dst_arr.ctypes.data, None)
        else:
            # Allocate a buffer and return a context manager
            # which flushes the buffer
            buf = np.empty(dst_arr.shape, dst_arr.dtype.newbyteorder('='))
            def buffer_flush():
                dst_arr[...] = buf
            return buffered_ptr_ctxmgr(buf.ctypes.data, buffer_flush)

class NumPyElementReadIter(IElementReadIter):
    def __init__(self, npyarr):
        if npyarr.ndim <= 0:
            raise IndexError('Need at least one dimension for iteration')
        self._index = 0
        self._len = len(npyarr)
        self._dshape = datashape.from_numpy(npyarr.shape[1:], npyarr.dtype)
        self.npyarr = npyarr

    @property
    def dshape(self):
        return self._dshape

    def __len__(self):
        return self._len

    def __next__(self):
        if self._index < self._len:
            i = self._index
            self._index = i + 1
            x = self.npyarr[i]
            # Make it C-contiguous and in native byte order
            x = np.ascontiguousarray(x, dtype=x.dtype.newbyteorder('='))
            self._tmpbuffer = x
            return x.ctypes.data
        else:
            raise StopIteration

class NumPyElementWriteIter(IElementWriteIter):
    def __init__(self, npyarr):
        if npyarr.ndim <= 0:
            raise IndexError('Need at least one dimension for iteration')
        self._index = 0
        self._len = len(npyarr)
        self._dshape = datashape.from_numpy(npyarr.shape[1:], npyarr.dtype)
        self._usebuffer = not (npyarr[0].flags.c_contiguous and npyarr.dtype.isnative)
        self._buffer_index = -1
        self._buffer = None
        self._buffer_index = -1
        self.npyarr = npyarr

    @property
    def dshape(self):
        return self._dshape

    def __len__(self):
        return self._len

    def flush(self):
        if self._usebuffer and self._buffer_index >= 0:
            self.npyarr[self._buffer_index] = self._buffer
            self._buffer_index = -1

    def __next__(self):
        # Copy the previous element to the array if it is buffered
        self.flush()
        if self._index < self._len:
            i = self._index
            self._index = i + 1
            if self._usebuffer:
                if self._buffer is None:
                    self._buffer = np.empty(self.npyarr.shape[1:], self.npyarr.dtype.newbyteorder('='))
                self._buffer_index = i
                return self._buffer.ctypes.data
            else:
                return self.npyarr.ctypes.data + self.npyarr.strides[0] * i
        else:
            raise StopIteration

    def close(self):
        self.flush()

class NumPyDataDescriptor(IDataDescriptor):
    """
    A Blaze data descriptor which exposes a NumPy array.
    """
    def __init__(self, npyarr):
        if not isinstance(npyarr, np.ndarray):
            raise TypeError('object is not a numpy array, has type %s' %
                            type(npyarr))
        self.npyarr = npyarr
        self._dshape = datashape.from_numpy(
            self.npyarr.shape, self.npyarr.dtype)

    @property
    def dshape(self):
        return self._dshape

    @property
    def writable(self):
        return self.npyarr.flags.writeable

    @property
    def immutable(self):
        # NumPy arrays lack an immutability concept
        return False

    @property
    def shape(self):
        return self._dshape.shape

    @property
    def nd(self):
        return len(self._dshape.shape)

    def __len__(self):
        if self.npyarr.ndim > 0:
            return self.npyarr.shape[0]
        else:
            raise IndexError('Cannot get the length of a zero-dimensional array')

    def __getitem__(self, key):
        # Just integer indices (no slices) for now
        if not isinstance(key, tuple):
            key = (key,)
        key = tuple([operator.index(i) for i in key])
        if len(key) == self.npyarr.ndim:
            return NumPyDataDescriptor(
                self.npyarr[...,np.newaxis][key].reshape(()))
        else:
            return NumPyDataDescriptor(self.npyarr[key])

    def __setitem__(self, key, value):
        self.npyarr.__setitem__(key, value)

    def __iter__(self):
        return numpy_descriptor_iter(self.npyarr)

    def element_reader(self, nindex):
        return NumPyElementReader(self.npyarr, nindex)

    def element_read_iter(self):
        return NumPyElementReadIter(self.npyarr)

    def element_writer(self, nindex):
        if self.writable:
            return NumPyElementWriter(self.npyarr, nindex)
        else:
            raise ArrayWriteError('Cannot write to readonly NumPy array')

    def element_write_iter(self):
        if self.writable:
            return contextlib.closing(NumPyElementWriteIter(self.npyarr))
        else:
            raise ArrayWriteError('Cannot write to readonly NumPy array')
