from __future__ import absolute_import
import operator
import bisect

from blaze.datashape import dshape
from . import (IElementReader, IElementWriter,
                IElementReadIter, IElementWriteIter,
                IDataDescriptor)

def cat_descriptor_iter(ddlist):
    for i, dd in enumerate(ddlist):
        for el in dd:
            yield el

class CatElementReader(IElementReader):
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

class CatElementReadIter(IElementReadIter):
    def __init__(self, catdd):
        assert catdd.ndim > 0
        self._catdd = catdd
        self._index = 0
        self._len = self._catdd.shape[0]

    def __len__(self):
        return self._len

    def __next__(self):
        raise NotImplemented

class CatDataDescriptor(IDataDescriptor):
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
            if not isinstance(dd, IDataDescriptor):
                raise ValueError('Provided ddlist has an element '
                                'which is not a data descriptor')
        self._ddlist = ddlist
        self._dshape = ds.cat_dshapes([dd.dshape for dd in ddlist])
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

    def __len__(self):
        return self._boundary_index[-1]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        # Just integer indices (no slices) for now
        boundary_index = self._boundary_index
        dim_size = boundary_index[-1]
        idx0 = operator.index(key[0])
        # Determine which data descriptor in the list to use
        if idx0 >= 0:
            if idx0 >= dim_size:
                raise IndexError(('Index %d is out of range '
                                'in dimension sized %d') % (idx0, dim_size))
        else:
            if idx0 < -dim_size:
                raise IndexError(('Index %d is out of range '
                                'in dimension sized %d') % (idx0, dim_size))
            idx0 += dim_size
        i = bisect.bisect_right(boundary_index, idx0) - 1
        # Call the i-th data descriptor to get the result
        return self._ddlist[i][(idx0 - boundary_index[i],) + key[1:]]

    def __iter__(self):
        return cat_descriptor_iter(self._ddlist)

    def element_reader(self, nindex):
        return CatElementReader(self, nindex)

    def element_read_iter(self):
        return CatElementReadIter(self)

