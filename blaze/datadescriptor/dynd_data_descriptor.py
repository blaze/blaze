from __future__ import absolute_import
import operator

from . import (IElementReader, IElementWriter,
                IElementReadIter, IElementWriteIter,
                IDataDescriptor)
from .. import datashape
from ..datashape import dshape

try:
    import dynd
    from dynd import nd, ndt, lowlevel
except ImportError:
    dynd = None

def dynd_descriptor_iter(dyndarr):
    for el in dyndarr:
        yield DyNDDataDescriptor(el)

class DyNDElementReader(IElementReader):
    def __init__(self, dyndarr, nindex):
        if nindex > dyndarr.undim:
            raise IndexError('Cannot have more indices than dimensions')
        self._nindex = nindex
        self._dshape = datashape.dshape(dyndarr.dshape).subarray(nindex)
        self._c_dtype = nd.dtype(str(self._dshape))
        self.dyndarr = dyndarr

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
        x = self.dyndarr[idx]
        # Make it C-contiguous and in native byte order
        x = x.cast(self._c_dtype)
        self._tmpbuffer = x
        return lowlevel.data_address_of(x)

class DyNDElementReadIter(IElementReadIter):
    def __init__(self, dyndarr):
        if dyndarr.undim <= 0:
            raise IndexError('Need at least one dimension for iteration')
        self._index = 0
        self._len = len(dyndarr)
        self._dshape = datashape.dshape(dyndarr.dshape)
        self._c_dtype = nd.dtype(str(self._dshape))
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
            x = x.cast(self._c_dtype)
            self._tmpbuffer = x
            return lowlevel.data_address_of(x)
        else:
            raise StopIteration

class DyNDDataDescriptor(IDataDescriptor):
    """
    A Blaze data descriptor which exposes a DyND array.
    """
    def __init__(self, dyndarr):
        if dynd is None:
            raise ImportError('dynd is not installed, all dynd '
                            'functionality is disabled')
        if not isinstance(dyndarr, nd.ndobject):
            raise TypeError('object is not a dynd array, has type %s' %
                            type(dyndarr))
        self.dyndarr = dyndarr
        self._dshape = dshape(dyndarr.dshape)

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

    def __iter__(self):
        return dynd_descriptor_iter(self.dyndarr)

    def element_reader(self, nindex):
        return DyNDElementReader(self.dyndarr, nindex)

    def element_read_iter(self):
        return DyNDElementReadIter(self.dyndarr)
