from __future__ import absolute_import
import operator

from blaze import dshape
from blaze.datashape import coretypes
from . import DataDescriptor, IGetDescriptor, \
                IDescriptorIter, IGetElement, IElementIter
import numpy as np

class NumPyGetDescriptor(IGetDescriptor):
    def __init__(self, npyarr, nindex):
        if nindex > npyarr.ndim:
            raise IndexError('Cannot have more indices than dimensions')
        self._nindex = nindex
        self.npyarr = npyarr

    @property
    def nindex(self):
        return self._nindex

    def get(self, idx):
        if len(idx) != self.nindex:
            raise IndexError('Incorrect number of indices (got %d, require %d)' %
                           (len(idx), self.nindex))
        idx = tuple([operator.index(i) for i in idx])
        return NumPyDataDescriptor(self.npyarr[idx])

class NumPyDescriptorIter(IDescriptorIter):
    def __init__(self, npyarr):
        if npyarr.ndim <= 0:
            raise IndexError('Need at least one dimension for iteration')
        self.npyarr = npyarr
        self._index = 0
        self._len = self.npyarr.shape[0]

    def __len__(self):
        return self._len

    def __next__(self):
        if self._index < self._len:
            i = self._index
            self._index = i + 1
            return NumPyDataDescriptor(self.npyarr[i])
        else:
            raise StopIteration

class NumPyGetElement(IGetElement):
    def __init__(self, npyarr, nindex):
        if nindex > npyarr.ndim:
            raise IndexError('Cannot have more indices than dimensions')
        self._nindex = nindex
        self.npyarr = npyarr

    @property
    def nindex(self):
        return self._nindex

    def get(self, idx):
        if len(idx) != self.nindex:
            raise IndexError('Incorrect number of indices (got %d, require %d)' %
                           (len(idx), self.nindex))
        idx = tuple([operator.index(i) for i in idx])
        x = self.npyarr[idx]
        # Make it C-contiguous and in native byte order
        x = np.ascontiguousarray(x, dtype=x.dtype.newbyteorder('='))
        self._tmpbuffer = x
        import ctypes
        return x.data_as(ctypes.c_void_p)

class NumPyElementIter(IElementIter):
    def __init__(self, npyarr):
        if npyarr.ndim <= 0:
            raise IndexError('Need at least one dimension for iteration')
        self.npyarr = npyarr
        self._index = 0
        self._len = self.npyarr.shape[0]

    def __len__(self):
        return self._len

    def __next__(self):
        if self._index < self._len:
            i = self._index
            self._index = i + 1
            x = self.npyarr[idx]
            # Make it C-contiguous and in native byte order
            x = np.ascontiguousarray(x, dtype=x.dtype.newbyteorder('='))
            self._tmpbuffer = x
            import ctypes
            return x.data_as(ctypes.c_void_p)
        else:
            raise StopIteration

class NumPyDataDescriptor(DataDescriptor):
    """
    A Blaze data descriptor which exposes a NumPy array.
    """
    def __init__(self, npyarr):
        self.npyarr = npyarr
        self._dshape = coretypes.from_numpy(self.npyarr.shape, self.npyarr.dtype)

    @property
    def dshape(self):
        return self._dshape

    def get_descriptor_interface(self, nindex):
        return NumPyGetDescriptor(self.npyarr, nindex)

    def descriptor_iter_interface(self):
        return NumPyDescriptorIter(self.npyarr)

    def get_element_interface(self, nindex):
        return NumPyGetElement(self.npyarr, nindex)

    def element_iter_interface(self):
        return NumPyElementIter(self.npyarr)
