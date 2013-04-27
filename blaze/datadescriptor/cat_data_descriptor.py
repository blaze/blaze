from __future__ import absolute_import
import operator

from blaze import dshape
from blaze.datashape import coretypes
from . import DataDescriptor, IGetDescriptor, \
                IDescriptorIter, IGetElement, IElementIter

class CatGetDescriptor(IGetDescriptor):
    def __init__(self, npyarr, nindex):
        assert nindex <= npyarr.ndim
        self._nindex = nindex
        self.npyarr = npyarr

    @property
    def nindex(self):
        return self._nindex

    def get(self, idx):
        assert len(idx) == self.nindex
        idx = tuple([operator.index(i) for i in idx])
        return CatDataDescriptor(self.npyarr[idx])

class CatDescriptorIter(IDescriptorIter):
    def __init__(self, npyarr):
        assert npyarr.ndim > 0
        self.npyarr = npyarr
        self._index = 0
        self._len = self.npyarr.shape[0]

    def __len__(self):
        return self._len

    def __next__(self):
        if self._index < self._len:
            i = self._index
            self._index = i + 1
            return CatDataDescriptor(self.npyarr[i])
        else:
            raise StopIteration

class CatGetElement(IGetElement):
    def __init__(self, npyarr, nindex):
        assert nindex <= npyarr.ndim
        self._nindex = nindex
        self.npyarr = npyarr

    @property
    def nindex(self):
        return self._nindex

    def get(self, idx):
        raise NotImplemented

class CatElementIter(IElementIter):
    def __init__(self, npyarr):
        assert npyarr.ndim > 0
        self.npyarr = npyarr
        self._index = 0
        self._len = self.npyarr.shape[0]

    def __len__(self):
        return self._len

    def __next__(self):
        raise NotImplemented

class CatDataDescriptor(DataDescriptor):
    """
    A Blaze data descriptor which concatenates a list
    of data descriptors, all of which have the same
    dshape after the first dimension.

    This presently doesn't support leading dimensions
    whose size is unknown (i.e. streaming dimensions).
    """
    def __init__(self, ddlist):
        if len(ddlist) <= 1:
            raise ValueError('Need at least 2 data descriptors to concatenate')
        for dd in ddlist:
            if not isinstance(dd, DataDescriptor):
                raise ValueError('Provided ddlist has an element which is not a data descriptor')
        self.ddlist = ddlist
        self._dshape = coretypes.cat_dshapes([dd.dshape for dd in ddlist])
        outer_dim_size = operator.index(self.ddlist[0][0])
        outer_dim_sizelist = [outer_dim_size]
        self._outer_dim_size = outer_dim_size
        self._outer_dim_sizelist = outer_dim_sizelist
        print 'cat dshape: ', self._dshape
 
    @property
    def dshape(self):
        return self._dshape

    def get_descriptor_interface(self, nindex):
        return CatGetDescriptor(self, nindex)

    def descriptor_iter_interface(self):
        return CatDescriptorIter(self)

    def get_element_interface(self, nindex):
        return CatGetElement(self, nindex)

    def element_iter_interface(self):
        return CatElementIter(self)

