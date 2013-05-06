from __future__ import absolute_import
import operator

from . import DataDescriptor, IGetElement, IElementIter
#from ..datashape import dshape  # apparently this is not used
import numpy as np
import blz

def blz_descriptor_iter(blzarr):
    if blzarr.ndim > 1:
        for el in blzarr:
            yield BLZDataDescriptor(el)
    else:
        for i in range(blzarr.shape[0]):
            # BLZ doesn't have a convenient way to avoid collapsing
            # to a scalar, this is a way to avoid that
            el = np.array(blzarr[i], dtype=blzarr.dtype)
            yield BLZDataDescriptor(el)

class BLZGetElement(IGetElement):
    def __init__(self, blzarr, nindex):
        if nindex > blzarr.ndim:
            raise IndexError('Cannot have more indices than dimensions')
        self._nindex = nindex
        self.blzarr = blzarr

    @property
    def nindex(self):
        return self._nindex

    def get(self, idx):
        if len(idx) != self.nindex:
            raise IndexError('Incorrect number of indices (got %d, require %d)' %
                           (len(idx), self.nindex))
        idx = tuple([operator.index(i) for i in idx])
        x = self.blzarr[idx]
        # x is already well-behaved (C-contiguous and native order)
        self._tmpbuffer = x
        import ctypes
        return x.data_as(ctypes.c_void_p)

class BLZElementIter(IElementIter):
    def __init__(self, blzarr):
        if blzarr.ndim <= 0:
            raise IndexError('Need at least one dimension for iteration')
        self.blzarr = blzarr
        self._index = 0
        self._len = self.blzarr.shape[0]

    def __len__(self):
        return self._len

    def __next__(self):
        if self._index < self._len:
            i = self._index
            self._index = i + 1
            x = self.blzarr[idx]
            # x is already well-behaved (C-contiguous and native order)
            self._tmpbuffer = x
            import ctypes
            return x.data_as(ctypes.c_void_p)
        else:
            raise StopIteration

class BLZDataDescriptor(DataDescriptor):
    """
    A Blaze data descriptor which exposes a BLZ array.
    """
    def __init__(self, blzarr):
        if not isinstance(blzarr, blz.barray):
            raise TypeError(
                'object is not a blz array, has type %s' % type(blzarr))
        self.blzarr = blzarr
        self._dshape = datashape.from_numpy(
            self.blzarr.shape, self.blzarr.dtype)

    @property
    def dshape(self):
        return self._dshape

    def __len__(self):
        if self.blzarr.ndim > 0:
            return self.blzarr.shape[0]
        else:
            raise IndexError('Cannot get the length of a zero-dimensional array')

    def __getitem__(self, key):
        # Just integer indices (no slices) for now
        if not isinstance(key, tuple):
            key = (key,)
        key = tuple([operator.index(i) for i in key])
        if len(key) == self.blzarr.ndim:
            return BLZDataDescriptor(np.array(blzarr[i], dtype=blzarr.dtype))
        else:
            return BLZDataDescriptor(self.blzarr[key])

    def __iter__(self):
        return blz_descriptor_iter(self.blzarr)

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

    def get_element_interface(self, nindex):
        return BLZGetElement(self.blzarr, nindex)

    def element_iter_interface(self):
        return BLZElementIter(self.blzarr)
