from __future__ import absolute_import, division, print_function

import os
import numpy as np
from dynd import nd
import datashape
import h5py

from . import DDesc, Capabilities
from .dynd_data_descriptor import DyND_DDesc
from .stream_data_descriptor import Stream_DDesc
from ..optional_packages import tables_is_here
if tables_is_here:
    import tables as tb



class HDF5_DDesc(DDesc):
    """
    A Blaze data descriptor which exposes a HDF5 dataset.
    """

    def __init__(self, path, datapath, mode='r', filters=None):
        self.path = path
        self.datapath = datapath
        self.mode = mode
        self.filters = filters

    @property
    def dshape(self):
        # This cannot be cached because the Array can change the dshape
        with tb.open_file(self.path, mode='r') as f:
            dset = f.get_node(self.datapath)
            odshape = datashape.from_numpy(dset.shape, dset.dtype)
        return odshape

    @property
    def capabilities(self):
        """The capabilities for the HDF5 arrays."""
        with tb.open_file(self.path, mode='r') as f:
            dset = f.get_node(self.datapath)
            appendable = isinstance(dset, (tb.EArray, tb.Table))
            queryable = isinstance(dset, (tb.Table,))
        caps = Capabilities(
            # HDF5 arrays can be updated
            immutable = False,
            # HDF5 arrays are concrete
            deferred = False,
            # HDF5 arrays are persistent
            persistent = True,
            # HDF5 arrays can be appended efficiently (EArrays and Tables)
            appendable = appendable,
            # PyTables Tables can be queried efficiently
            queryable = queryable,
            remote = False,
            )
        return caps

    def dynd_arr(self):
        # Positionate at the beginning of the file
        with tb.open_file(self.path, mode='r') as f:
            dset = f.get_node(self.datapath)
            dset = nd.array(dset[:], dtype=dset.dtype)
        return dset

    def __array__(self):
        with tb.open_file(self.path, mode='r') as f:
            dset = f.get_node(self.datapath)
            dset = dset[:]
        return dset

    def __len__(self):
        with tb.open_file(self.path, mode='r') as f:
            dset = f.get_node(self.datapath)
            arrlen = len(dset)
        return arrlen

    def __getitem__(self, key):
        with tb.open_file(self.path, mode='r') as f:
            dset = f.get_node(self.datapath)
            # The returned arrays are temporary buffers,
            # so must be flagged as readonly.
            dyndarr = nd.asarray(dset[key], access='readonly')
        return DyND_DDesc(dyndarr)

    def __setitem__(self, key, value):
        # HDF5 arrays can be updated
        with tb.open_file(self.path, mode=self.mode) as f:
            dset = f.get_node(self.datapath)
            dset[key] = value

    def __iter__(self):
        f = tb.open_file(self.path, mode='r')
        dset = f.get_node(self.datapath)
        # Get rid of the leading dimension on which we iterate
        dshape = datashape.from_numpy(dset.shape[1:], dset.dtype)
        for el in dset:
            if hasattr(el, "nrow"):
                yield DyND_DDesc(nd.array(el[:], type=str(dshape)))
            else:
                yield DyND_DDesc(nd.array(el, type=str(dshape)))
        dset._v_file.close()

    def where(self, condition):
        """Iterate over values fulfilling a condition."""
        f = tb.open_file(self.path, mode='r')
        dset = f.get_node(self.datapath)
        # Get rid of the leading dimension on which we iterate
        dshape = datashape.from_numpy(dset.shape[1:], dset.dtype)
        for el in dset.where(condition):
            yield DyND_DDesc(nd.array(el[:], type=str(dshape)))
        dset._v_file.close()

    def getattr(self, name):
        with tb.open_file(self.path, mode=self.mode) as f:
            dset = f.get_node(self.datapath)
            if hasattr(dset, 'cols'):
                return DyND_DDesc(
                    nd.asarray(getattr(dset.cols, name)[:],
                               access='readonly'))
            else:
                raise IndexError("not an HDF5 compound dataset")

    def iterchunks(self, blen=100):
        with h5py.File(self.path, mode='r') as f:
            arr = f[self.datapath]
            for i in range(0, arr.shape[0], blen):
                yield DyND_DDesc(nd.asarray(np.array(arr[i:i+blen]),
                                 access='readonly'))

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
        with tb.open_file(self.path, mode=self.mode) as f:
            dset = f.get_node(self.datapath)
            dset.append(values_arr.reshape(shape_vals))

    def remove(self):
        """Remove the persistent storage."""
        os.unlink(self.path)
