from __future__ import absolute_import
import operator

from blaze import dshape, datashape
from . import DataDescriptor, IGetElement, IElementIter
import cffi

def cffi_descriptor_iter(npyarr):
    if npyarr.ndim > 1:
        for el in npyarr:
            yield CFFIDataDescriptor(el)
    else:
        for i in range(npyarr.shape[0]):
            # CFFI doesn't have a convenient way to avoid collapsing
            # to a scalar, this is a way to avoid that
            el = npyarr[...,np.newaxis][i].reshape(())
            yield CFFIDataDescriptor(el)

class CFFIGetElement(IGetElement):
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

class CFFIElementIter(IElementIter):
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

class CFFIDataDescriptor(DataDescriptor):
    """
    A Blaze data descriptor which exposes CFFI memory.
    """
    def __init__(self, ffi, cdata):
        """
        Parameters
        ----------
        ffi : cffi.FFI
            The cffi namespace which contains the cdata.
        cdata : cffi.CData
            The cffi data.
        """
        if not isinstance(ffi, cffi.FFI):
            raise TypeError('object is not a cffi.FFI object, has type %s' % type(ffi))
        if not isinstance(cdata, cffi.CData):
            raise TypeError('object is not a cffi.CData object, has type %s' % type(cdata))
        self.ffi = ffi
        self.cdata = cdata
        self.ctype = ffi.typeof(cdata)
        self._dshape = datashape.from_cffi(self.ctype)
        if isinstance(self._dshape, DataShape) and \
                        isinstance(self._dshape[0], TypeVar):
            # If the outermost dimension is an array without fixed
            # size, get the size from the data
            self._dshape = DataShape([Fixed(len(cdata))] + self._dshape[1:])

    @property
    def dshape(self):
        return self._dshape

    def __len__(self):
        if isinstance(self._dshape, DataShape):
            return operator.index(self._dshape[0])
        else:
            raise IndexError('Cannot get the length of a zero-dimensional array')

    def __getitem__(self, key):
        # Just integer indices (no slices) for now
        if not isinstance(key, tuple):
            key = (key,)
        key = tuple([operator.index(i) for i in key])
        if len(key) == self.npyarr.ndim:
            return CFFIDataDescriptor(self.npyarr[...,np.newaxis][key].reshape(()))
        else:
            return CFFIDataDescriptor(self.npyarr[key])

    def __iter__(self):
        return cffi_descriptor_iter(self.npyarr)

    def get_element_interface(self, nindex):
        return CFFIGetElement(self.npyarr, nindex)

    def element_iter_interface(self):
        return CFFIElementIter(self.npyarr)

