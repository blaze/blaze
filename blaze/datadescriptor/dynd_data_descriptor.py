from __future__ import absolute_import
import operator
import contextlib

from .data_descriptor import (IElementReader, IElementWriter,
                IElementReadIter, IElementWriteIter,
                IDataDescriptor, buffered_ptr_ctxmgr)
from .. import datashape
from ..datashape import dshape

from dynd import nd, ndt, _lowlevel

def dynd_descriptor_iter(dyndarr):
    for el in dyndarr:
        yield DyNDDataDescriptor(el)

class DyNDElementReader(IElementReader):
    def __init__(self, dyndarr, nindex):
        if nindex > nd.ndim_of(dyndarr):
            raise IndexError('Cannot have more indices than dimensions')
        self._nindex = nindex
        self._dshape = datashape.dshape(nd.dshape_of(dyndarr)).subarray(nindex)
        self._c_dtype = ndt.type(str(self._dshape))
        self.dyndarr = dyndarr

    @property
    def dshape(self):
        return self._dshape

    @property
    def nindex(self):
        return self._nindex

    def read_single(self, idx, count=1):
        idxlen = len(idx)
        if idxlen != self.nindex:
            raise IndexError('Incorrect number of indices (got %d, require %d)' %
                           (len(idx), self.nindex))
        idx = tuple([operator.index(i) for i in idx])
        if count == 1:
            x = self.dyndarr[idx]
            # Make it C-contiguous and in native byte order
            x = x.cast(self._c_dtype).eval()
            self._tmpbuffer = x
            return _lowlevel.data_address_of(x)
        else:
            idx = idx[:-1] + (slice(idx[-1], idx[-1]+count),)
            x = self.dyndarr[idx]
            # Make it C-contiguous and in native byte order
            x = x.cast(ndt.make_fixed_dim_dtype(count, self._c_dtype)).eval()
            self._tmpbuffer = x
            return _lowlevel.data_address_of(x)

class DyNDElementWriter(IElementWriter):
    def __init__(self, dyndarr, nindex):
        if nindex > nd.ndim_of(dyndarr):
            raise IndexError('Cannot have more indices than dimensions')
        self._nindex = nindex
        self._dshape = datashape.dshape(nd.dshape_of(dyndarr)).subarray(nindex)
        self._c_dtype = ndt.type(str(self._dshape))
        self.dyndarr = dyndarr

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
        # Create a temporary DyND array around the ptr data.
        # Note that we can't provide an owner because the parameter
        # is just the bare pointer.
        tmp = _lowlevel.py_api.array_from_ptr(self._c_dtype, ptr,
                        None, 'readonly')
        # Use DyND's assignment to set the values
        self.dyndarr[idx] = tmp

    def buffered_ptr(self, idx):
        if len(idx) != self.nindex:
            raise IndexError('Incorrect number of indices (got %d, require %d)' %
                           (len(idx), self.nindex))
        dst_arr = self.dyndarr[idx]
        buf_arr = dst_arr.cast(self._c_dtype).eval()
        buf_ptr = _lowlevel.data_address_of(buf_arr)
        if buf_ptr == _lowlevel.data_address_of(dst_arr):
            # If no buffering is needed, just return the pointer
            return buffered_ptr_ctxmgr(buf_ptr, None)
        else:
            def buffer_flush():
                dst_arr[...] = buf_arr
            return buffered_ptr_ctxmgr(buf_ptr, buffer_flush)

class DyNDElementReadIter(IElementReadIter):
    def __init__(self, dyndarr):
        if nd.ndim_of(dyndarr) <= 0:
            raise IndexError('Need at least one dimension for iteration')
        self._index = 0
        self._len = len(dyndarr)
        self._dshape = datashape.dshape(nd.dshape_of(dyndarr)).subarray(1)
        self._c_dtype = ndt.type(str(self._dshape))
        self.dyndarr = dyndarr

    @property
    def dshape(self):
        return self._dshape

    def __len__(self):
        return self._len

    def __next__(self):
        if self._index < self._len:
            i = self._index
            self._index = i + 1
            x = self.dyndarr[i]
            # Make it C-contiguous and in native byte order
            x = x.cast(self._c_dtype).eval()
            self._tmpbuffer = x
            return _lowlevel.data_address_of(x)
        else:
            raise StopIteration

class DyNDElementWriteIter(IElementWriteIter):
    def __init__(self, dyndarr):
        if nd.ndim_of(dyndarr) <= 0:
            raise IndexError('Need at least one dimension for iteration')
        self._index = 0
        self._len = len(dyndarr)
        ds = datashape.dshape(nd.dshape_of(dyndarr))
        self._dshape = ds.subarray(1)
        self._c_dtype = ndt.type(str(self._dshape))
        self._usebuffer = (ndt.type(str(ds)) != nd.type_of(dyndarr))
        self._buffer = None
        self._buffer_index = -1
        self.dyndarr = dyndarr

    @property
    def dshape(self):
        return self._dshape

    def __len__(self):
        return self._len

    def flush(self):
        if self._usebuffer and self._buffer_index >= 0:
            self.dyndarr[self._buffer_index] = self._buffer
            self._buffer_index = -1

    def __next__(self):
        # Copy the previous element to the array if it is buffered
        self.flush()
        if self._index < self._len:
            i = self._index
            self._index = i + 1
            if self._usebuffer:
                if self._buffer is None:
                    self._buffer = nd.empty(self._c_dtype)
                self._buffer_index = i
                return _lowlevel.data_address_of(self._buffer)
            else:
                return _lowlevel.data_address_of(self.dyndarr[i])
        else:
            raise StopIteration

    def close(self):
        self.flush()

class DyNDDataDescriptor(IDataDescriptor):
    """
    A Blaze data descriptor which exposes a DyND array.
    """
    def __init__(self, dyndarr):
        if not isinstance(dyndarr, nd.array):
            raise TypeError('object is not a dynd array, has type %s' %
                            type(dyndarr))
        self.dyndarr = dyndarr
        self._dshape = dshape(nd.dshape_of(dyndarr))

    @property
    def dshape(self):
        return self._dshape

    @property
    def writable(self):
        return self.dyndarr.access_flags == 'readwrite'

    @property
    def immutable(self):
        return self.dyndarr.access_flags == 'immutable'

    def __len__(self):
        return len(self.dyndarr)

    def __getitem__(self, key):
        return DyNDDataDescriptor(self.dyndarr[key])

    def __setitem__(self, key, value):
        # TODO: This is a horrible hack, we need to specify item setting
        #       via well-defined interfaces, not punt to another system.
        self.dyndarr.__setitem__(key, value)

    def __iter__(self):
        return dynd_descriptor_iter(self.dyndarr)

    def element_reader(self, nindex):
        return DyNDElementReader(self.dyndarr, nindex)

    def element_read_iter(self):
        return DyNDElementReadIter(self.dyndarr)

    def element_writer(self, nindex):
        if self.writable:
            return DyNDElementWriter(self.dyndarr, nindex)
        else:
            raise ArrayWriteError('Cannot write to readonly DyND array')

    def element_write_iter(self):
        if self.writable:
            return contextlib.closing(DyNDElementWriteIter(self.dyndarr))
        else:
            raise ArrayWriteError('Cannot write to readonly DyND array')
