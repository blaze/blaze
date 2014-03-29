from __future__ import absolute_import, division, print_function

import numpy as np
from dynd import nd
import datashape

from . import IDataDescriptor, Capabilities
from ..optional_packages import tables_is_here
if tables_is_here:
    import tables as tb
from .dynd_data_descriptor import DyNDDataDescriptor


# WARNING!  PyTables always return NumPy arrays when doing indexing
# operations.  This is why DyNDDataDescriptor is used for returning
# the values here.
def hdf5_descriptor_iter(h5arr):
    for i in range(len(h5arr)):
        # PyTables doesn't have a convenient way to avoid collapsing
        # to a scalar, this is a way to avoid that
        el = np.array(h5arr[i], dtype=h5arr.dtype)
        yield DyNDDataDescriptor(nd.array(el))
    h5arr._v_file.close()


class HDF5DataDescriptor(IDataDescriptor):
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
            h5arr = f.get_node(f.root, self.datapath)
            odshape = datashape.from_numpy(h5arr.shape, h5arr.dtype)
        return odshape

    @property
    def capabilities(self):
        """The capabilities for the HDF5 arrays."""
        with tb.open_file(self.path, mode='r') as f:
            h5arr = f.get_node(f.root, self.datapath)
            appendable = isinstance(h5arr, (tb.EArray, tb.Table)),
        caps = Capabilities(
            # HDF5 arrays can be updated
            immutable = False,
            # HDF5 arrays are concrete
            deferred = False,
            # HDF5 arrays are persistent
            persistent = True,
            # HDF5 arrays can be appended efficiently (EArrays and Tables)
            appendable = appendable,
            remote = False,
            )
        return caps

    def dynd_arr(self):
        # Positionate at the beginning of the file
        with tb.open_file(self.path, mode='r') as f:
            h5arr = f.get_node(f.root, self.datapath)
            h5arr = nd.array(h5arr[:], dtype=h5arr.dtype)
        return h5arr

    def __array__(self):
        with tb.open_file(self.path, mode='r') as f:
            h5arr = f.get_node(f.root, self.datapath)
            h5arr = h5arr[:]
        return h5arr

    def __len__(self):
        with tb.open_file(self.path, mode='r') as f:
            h5arr = f.get_node(f.root, self.datapath)
            arrlen = len(h5arr)
        return arrlen

    def __getitem__(self, key):
        with tb.open_file(self.path, mode='r') as f:
            h5arr = f.get_node(f.root, self.datapath)
            # The returned arrays are temporary buffers,
            # so must be flagged as readonly.
            dyndarr = nd.asarray(h5arr[key], access='readonly')
        return DyNDDataDescriptor(dyndarr)

    def __setitem__(self, key, value):
        # HDF5 arrays can be updated
        with tb.open_file(self.path, mode=self.mode) as f:
            h5arr = f.get_node(f.root, self.datapath)
            h5arr[key] = value

    def __iter__(self):
        f = tb.open_file(self.path, mode='r')
        h5arr = f.get_node(f.root, self.datapath)
        return hdf5_descriptor_iter(h5arr)

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
            h5arr = f.get_node(f.root, self.datapath)
            h5arr.append(values_arr.reshape(shape_vals))
