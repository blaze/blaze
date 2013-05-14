from __future__ import absolute_import
import operator

from . import (IElementReader, IElementWriter,
                IElementReadIter, IElementWriteIter,
                IDataDescriptor)
from .. import datashape
import numpy as np
from blaze import blz
from .numpy_data_descriptor import NumPyDataDescriptor

# WARNING!  BLZ always return NumPy arrays when doing indexing
# operations.  This is why NumPyDataDescriptor is used for returning
# the values here.  Ideally, BLZ should return pure buffers instead.

def blz_descriptor_iter(blzarr):
    if blzarr.ndim > 1:
        for el in blzarr:
            yield NumPyDataDescriptor(el)
    else:
        for i in range(blzarr.shape[0]):
            # BLZ doesn't have a convenient way to avoid collapsing
            # to a scalar, this is a way to avoid that
            el = np.array(blzarr[i], dtype=blzarr.dtype)
            yield NumPyDataDescriptor(el)

class BLZElementReader(IElementReader):
    def __init__(self, blzarr, nindex, ds):
        if nindex > blzarr.ndim:
            raise IndexError('Cannot have more indices than dimensions')
        self._nindex = nindex
        self.blzarr = blzarr
        self._dshape = ds

    @property
    def nindex(self):
        return self._nindex

    @property
    def dshape(self):
        return self._dshape

    def get(self, idx):
        if len(idx) != self.nindex:
            raise IndexError('Incorrect number of indices (got %d, require %d)' %
                           (len(idx), self.nindex))
        idx = tuple([operator.index(i) for i in idx])
        x = self.blzarr[idx]
        # x is already well-behaved (C-contiguous and native order)
        self._tmpbuffer = x
        return x.ctypes.data()

class BLZElementReadIter(IElementReadIter):
    def __init__(self, blzarr, ds):
        if blzarr.ndim <= 0:
            raise IndexError('Need at least one dimension for iteration')
        self.blzarr = blzarr
        self._index = 0
        self._len = self.blzarr.shape[0]
        self._dshape = ds

    @property
    def dshape(self):
        return self._dshape

    def __len__(self):
        return self._len

    def __next__(self):
        if self._index < self._len:
            i = self._index
            self._index = i + 1
            x = self.blzarr[i:i+1]
            # x is already well-behaved (C-contiguous and native order)
            self._tmpbuffer = x
            return x.ctypes.data
        else:
            raise StopIteration

class BLZDataDescriptor(IDataDescriptor):
    """
    A Blaze data descriptor which exposes a BLZ array.
    """
    def __init__(self, obj):
        # This is a low level interface, so strictly
        # require a BLZ barray here
        if not isinstance(obj, blz.barray):
            raise TypeError(('object is not a blz array, '
                             'it has type %r') % type(obj))
        self.blzarr = obj
        self._dshape = datashape.from_numpy(obj.shape, obj.dtype)

    @property
    def dshape(self):
        return self._dshape

    def __len__(self):
        # BLZ arrays are never scalars
        return self.blzarr.shape[0]

    def __getitem__(self, key):
        # Just integer indices (no slices) for now
        if not isinstance(key, tuple):
            key = (key,)
        key = tuple([operator.index(i) for i in key])
        blzarr = self.blzarr
        if len(key) == blzarr.ndim:
            return NumPyDataDescriptor(np.array(blzarr[i]))
        else:
            return NumPyDataDescriptor(blzarr[key])

    def __iter__(self):
        return blz_descriptor_iter(self.blzarr)

    def append(self, values):
        """Append a list of values."""
        obj = self.blzarr
        obj.append(values)
        obj.flush()  # XXX This should be in BLZ barray object instead?
        # Update the dshape
        self._dshape = datashape.from_numpy(obj.shape, obj.dtype)

    def iterchunks(self, blen=None, start=None, stop=None):
        """Return chunks of size `blen` (in leading dimension).

        Parameters
        ----------
        blen : int
            The length, in rows, of the buffers that are returned.
        start : int
            Where the iterator starts.  The default is to start at the
            beginning.
        stop : int
            Where the iterator stops. The default is to stop at the end.

        Returns
        -------
        out : iterable
            This iterable returns buffers as NumPy arays of
            homogeneous or structured types, depending on whether
            `self.original` is a barray or a btable object.

        See Also
        --------
        wherechunks

        """
        # Return the iterable
        return blz.iterblocks(self.blzarr, blen, start, stop)

    def wherechunks(self, expression, blen=None, outfields=None, limit=None,
                    skip=0):
        """Return chunks fulfilling `expression`.

        Iterate over the rows that fullfill the `expression` condition
        on Table `self.original` in blocks of size `blen`.

        Parameters
        ----------
        expression : string or barray
            A boolean Numexpr expression or a boolean barray.
        blen : int
            The length of the block that is returned.  The default is the
            chunklen, or for a btable, the minimum of the different column
            chunklens.
        outfields : list of strings or string
            The list of column names that you want to get back in results.
            Alternatively, it can be specified as a string such as 'f0 f1' or
            'f0, f1'.  If None, all the columns are returned.
        limit : int
            A maximum number of elements to return.  The default is return
            everything.
        skip : int
            An initial number of elements to skip.  The default is 0.

        Returns
        -------
        out : iterable
            This iterable returns buffers as NumPy arrays made of
            structured types (or homogeneous ones in case `outfields` is a
            single field.

        See Also
        --------
        iterchunks

        """
	# Return the iterable
        return blz.whereblocks(self.blzarr, expression, blen,
			       outfields, limit, skip)

    def element_reader(self, nindex):
        return BLZElementReader(self.blzarr, nindex, self.dshape.subarray(nindex))

    def element_read_iter(self):
        return BLZElementReadIter(self.blzarr, self.dshape.subarray(1))
