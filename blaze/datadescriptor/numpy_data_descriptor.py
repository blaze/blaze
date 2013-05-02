from __future__ import absolute_import
import operator

from . import DataDescriptor, IGetElement, IElementIter
from .. import datashape
from ..datashape import dshape
import numpy as np

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
        return x.ctypes.data_as(ctypes.c_void_p)

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
            x = self.npyarr[i]
            # Make it C-contiguous and in native byte order
            x = np.ascontiguousarray(x, dtype=x.dtype.newbyteorder('='))
            self._tmpbuffer = x
            import ctypes
            return x.ctypes.data_as(ctypes.c_void_p)
        else:
            raise StopIteration

class NumPyDataDescriptor(DataDescriptor):
    """
    A Blaze data descriptor which exposes a NumPy array.
    """
    def __init__(self, npyarr):
        if not isinstance(npyarr, np.ndarray):
            raise TypeError('object is not a numpy array, has type %s' %
                            type(npyarr))
        self.npyarr = npyarr
        self._dshape = datashape.from_numpy(self.npyarr.shape, self.npyarr.dtype)

    @property
    def dshape(self):
        return self._dshape

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
            return NumPyDataDescriptor(self.npyarr[...,np.newaxis][key].reshape(()))
        else:
            return NumPyDataDescriptor(self.npyarr[key])

    def __iter__(self):
        return numpy_descriptor_iter(self.npyarr)

    def get_element_interface(self, nindex):
        return NumPyGetElement(self.npyarr, nindex)

    def element_iter_interface(self):
        return NumPyElementIter(self.npyarr)
