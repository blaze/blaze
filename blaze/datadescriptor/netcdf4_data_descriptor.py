from __future__ import absolute_import, division, print_function

import os
import numpy as np
from dynd import nd
import datashape

from . import DDesc, Capabilities
from .dynd_data_descriptor import DyND_DDesc
from .stream_data_descriptor import Stream_DDesc
from ..optional_packages import netCDF4_is_here
if netCDF4_is_here:
    import netCDF4



class netCDF4_DDesc(DDesc):
    """
    A Blaze data descriptor which exposes a netCDF4 dataset.
    """

    def __init__(self, path, datapath, mode='r', filters=None):
        self.path = path
        self.datapath = datapath
        self.mode = mode
        self.filters = filters

    @property
    def dshape(self):
        # This cannot be cached because the Array can change the dshape
        with netCDF4.Dataset(self.path, mode='r') as f:
            dset = f.variables[self.datapath]
            odshape = datashape.from_numpy(dset.shape, dset.dtype)
        return odshape

    @property
    def capabilities(self):
        """The capabilities for the netCDF4 arrays."""
        with netCDF4.Dataset(self.path, mode='r') as f:
            dset = f.variables[self.datapath]
            appendable = isinstance(dset, netCDF4.Variable)
        caps = Capabilities(
            # netCDF4 arrays can be updated
            immutable = False,
            # netCDF4 arrays are concrete
            deferred = False,
            # netCDF4 arrays are persistent
            persistent = True,
            # netCDF4 arrays can be appended efficiently
            appendable = appendable,
            # netCDF4 arrays cannot be queried efficiently
            queryable = False,
            remote = False,
            )
        return caps

    def dynd_arr(self):
        # Positionate at the beginning of the file
        with netCDF4.Dataset(self.path, mode='r') as f:
            dset = f.variables[self.datapath]
            dset = nd.array(dset[:], dtype=dset.dtype)
        return dset

    def __array__(self):
        with netCDF4.Dataset(self.path, mode='r') as f:
            dset = f.variables[self.datapath]
            dset = dset[:]
        return dset

    def __len__(self):
        with netCDF4.Dataset(self.path, mode='r') as f:
            dset = f.variables[self.datapath]
            arrlen = len(dset)
        return arrlen

    def __getitem__(self, key):
        with netCDF4.Dataset(self.path, mode='r') as f:
            dset = f.variables[self.datapath]
            # The returned arrays are temporary buffers,
            # so must be flagged as readonly.
            dyndarr = nd.asarray(dset[key], access='readonly')
        return DyND_DDesc(dyndarr)

    def __setitem__(self, key, value):
        # netCDF4 arrays can be updated
        with netCDF4.Dataset(self.path, mode=self.mode) as f:
            dset = f.variables[self.datapath]
            dset[key] = value

    def __iter__(self):
        f = netCDF4.Dataset(self.path, mode='r')
        dset = f.variables[self.datapath]
        # Get rid of the leading dimension on which we iterate
        dshape = datashape.from_numpy(dset.shape[1:], dset.dtype)
        for el in dset:
            if hasattr(el, "nrow"):
                yield DyND_DDesc(nd.array(el[:], type=str(dshape)))
            else:
                yield DyND_DDesc(nd.array(el, type=str(dshape)))
        dset._v_file.close()

    def getattr(self, name):
        with netCDF4.Dataset(self.path, mode=self.mode) as f:
            dset = f.variables[self.datapath]
            if hasattr(dset, 'cols'):
                return DyND_DDesc(
                    nd.asarray(getattr(dset.cols, name)[:],
                               access='readonly'))
            else:
                raise IndexError("not an netCDF4 compound dataset")

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
        with netCDF4.Dataset(self.path, mode=self.mode) as f:
            dset = f.variables[self.datapath]
            dset[len(dset):] = values_arr.reshape(shape_vals)

    def remove(self):
        """Remove the persistent storage."""
        os.unlink(self.path)
