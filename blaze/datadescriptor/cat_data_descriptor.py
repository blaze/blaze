from __future__ import absolute_import
import operator
import bisect

from blaze import dshape
from blaze.datashape import coretypes
from . import DataDescriptor, IGetDescriptor, \
                IDescriptorIter, IGetElement, IElementIter

class CatGetDescriptor(IGetDescriptor):
    def __init__(self, catdd, nindex):
        if nindex > catdd._ndim:
            raise IndexError('Cannot have more indices than dimensions')
        self._nindex = nindex
        self._boundary_index = catdd._boundary_index
        self._gdlist = [dd.get_descriptor_interface(nindex) for dd in catdd._ddlist]

    @property
    def nindex(self):
        return self._nindex

    def get(self, idx):
        if len(idx) != self.nindex:
            raise IndexError('Incorrect number of indices (got %d, require %d)' %
                           (len(idx), self.nindex))
        boundary_index = self._boundary_index
        dim_size = boundary_index[-1]
        idx0 = operator.index(idx[0])
        # Determine which data descriptor in the list to use
        if idx0 >= 0:
            if idx0 >= dim_size:
                raise IndexError('Index %d is out of range in dimension sized %d' %
                                (idx0, dim_size))
        else:
            if idx0 < -dim_size:
                raise IndexError('Index %d is out of range in dimension sized %d' %
                                (idx0, dim_size))
            idx0 += dim_size
        i = bisect.bisect_right(boundary_index, idx0) - 1
        print('idx0: %d, i: %d, idx: %s' % (idx0, i, idx))
        # Call the i-th data descriptor to get the result
        return self._gdlist[i].get([idx0 - boundary_index[i]] + idx[1:])

class CatDescriptorIter(IDescriptorIter):
    def __init__(self, catdd):
        assert catdd.ndim > 0
        self._catdd = catdd
        self._index = 0
        self._len = self._catdd.shape[0]

    def __len__(self):
        return self._len

    def __next__(self):
        if self._index < self._len:
            i = self._index
            self._index = i + 1
            return CatDataDescriptor(self._catdd[i])
        else:
            raise StopIteration

class CatGetElement(IGetElement):
    def __init__(self, catdd, nindex):
        if nindex > catdd._ndim:
            raise IndexError('Cannot have more indices than dimensions')
        self._nindex = nindex
        self._catdd = catdd

    @property
    def nindex(self):
        return self._nindex

    def get(self, idx):
        raise NotImplemented

class CatElementIter(IElementIter):
    def __init__(self, catdd):
        assert catdd.ndim > 0
        self._catdd = catdd
        self._index = 0
        self._len = self._catdd.shape[0]

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
        self._ddlist = ddlist
        self._dshape = coretypes.cat_dshapes([dd.dshape for dd in ddlist])
        self._ndim = len(self._dshape[:]) - 1
        # Create a list of boundary indices
        boundary_index = [0]
        for dd in ddlist:
            dim_size = operator.index(dd.dshape[0])
            boundary_index.append(dim_size + boundary_index[-1])
        self._boundary_index = boundary_index
 
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

