from __future__ import absolute_import, division, print_function

import numpy as np
import blz
from dynd import nd
import datashape

from . import DDesc, Capabilities
from .dynd_data_descriptor import DyND_DDesc
from .stream_data_descriptor import Stream_DDesc
from shutil import rmtree


class BLZ_DDesc(DDesc):
    """
    A Blaze data descriptor which exposes a BLZ array.
    """
    def __init__(self, path=None, mode='r', **kwargs):
        self.path = path
        self.mode = mode
        self.kwargs = kwargs
        if isinstance(path, (blz.barray, blz.btable)):
            self.blzarr = path
            self.path = path.rootdir
        elif mode != 'w':
            self.blzarr = blz.open(rootdir=path, mode=mode, **kwargs)
        else:
            # This will be set in the constructor later on
            self.blzarr = None

    @property
    def dshape(self):
        # This cannot be cached because the BLZ can change the dshape
        obj = self.blzarr
        return datashape.from_numpy(obj.shape, obj.dtype)

    @property
    def capabilities(self):
        """The capabilities for the BLZ arrays."""
        if self.blzarr is None:
            persistent = False
        else:
            persistent = self.blzarr.rootdir is not None
        if isinstance(self.blzarr, blz.btable):
            queryable = True
        else:
            queryable = False
        return Capabilities(
            # BLZ arrays can be updated
            immutable = False,
            # BLZ arrays are concrete
            deferred = False,
            # BLZ arrays can be either persistent of in-memory
            persistent = persistent,
            # BLZ arrays can be appended efficiently
            appendable = True,
            # BLZ btables can be queried efficiently
            queryable = queryable,
            remote = False,
            )

    def __array__(self):
        return np.array(self.blzarr)

    def __len__(self):
        # BLZ arrays are never scalars
        return len(self.blzarr)

    def __getitem__(self, key):
        blzarr = self.blzarr
        # The returned arrays are temporary buffers,
        # so must be flagged as readonly.
        return DyND_DDesc(nd.asarray(blzarr[key], access='readonly'))

    def __setitem__(self, key, value):
        # We decided that BLZ should be read and append only
        raise NotImplementedError

    def __iter__(self):
        dset = self.blzarr
        # Get rid of the leading dimension on which we iterate
        dshape = datashape.from_numpy(dset.shape[1:], dset.dtype)
        for el in self.blzarr:
            yield DyND_DDesc(nd.array(el, type=str(dshape)))

    def where(self, condition, user_dict=None):
        """Iterate over values fulfilling a condition."""
        dset = self.blzarr
        # Get rid of the leading dimension on which we iterate
        dshape = datashape.from_numpy(dset.shape[1:], dset.dtype)
        for el in dset.where(condition):
            yield DyND_DDesc(nd.array(el, type=str(dshape)))

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
        return blz.whereblocks(self.blzarr, expression, blen, outfields,
                               limit, skip)


    def getattr(self, name):
        if isinstance(self.blzarr, blz.btable):
            return DyND_DDesc(nd.asarray(self.blzarr[name], access='readonly'))
        else:
            raise IndexError("not a btable BLZ dataset")

    # This is not part of the DDesc interface itself, but can
    # be handy for other situations not requering full compliance with
    # it.
    def append(self, values):
        """Append a list of values."""
        shape, dtype = datashape.to_numpy(self.dshape)
        values_arr = np.array(values, dtype=dtype)
        shape_vals = values_arr.shape
        if len(shape_vals) < len(shape):
            shape_vals = (1,) + shape_vals
        if len(shape_vals) != len(shape):
            raise ValueError("shape of values is not compatible")
        # Now, do the actual append
        self.blzarr.append(values_arr.reshape(shape_vals))
        self.blzarr.flush()

    def remove(self):
        """Remove the persistent storage."""
        if self.capabilities.persistent:
            rmtree(self.path)
