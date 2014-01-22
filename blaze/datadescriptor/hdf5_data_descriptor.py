from __future__ import absolute_import, division, print_function

import numpy as np
from dynd import nd
import datashape

from . import IDataDescriptor, Capabilities
import tables
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


class HDF5DataDescriptor(IDataDescriptor):
    """
    A Blaze data descriptor which exposes a HDF5 dataset.
    """
    def __init__(self, obj):
        # We are going to support both homogeneous and heterogeneous
        # datasets, but not VL types (VLArray) for the time being.
        if not isinstance(obj, (tb.Array, tb.Table)):
            raise TypeError(('object is not a supported HDF5 dataset, '
                             'it has type %r') % type(obj))
        self.h5arr = obj

    @property
    def dshape(self):
        # This cannot be cached because the Array can change the dshape
        obj = self.h5arr
        return datashape.from_numpy(obj.shape, obj.dtype)

    @property
    def capabilities(self):
        """The capabilities for the HDF5 arrays."""
        return Capabilities(
            # HDF5 arrays can be updated
            immutable = False,
            # HDF5 arrays are concrete
            deferred = False,
            # HDF5 arrays are persistent
            persistent = True,
            # HDF5 arrays can be appended efficiently for EArrays and Tables
            appendable = isinstance(self.h5arr, (tb.EArray, tb.Table),
            remote = False,
            )

    def __array__(self):
        return np.array(self.h5arr)

    def __len__(self):
        return len(self.h5arr)

    def __getitem__(self, key):
        h5arr = self.h5arr
        # The returned arrays are temporary buffers,
        # so must be flagged as readonly.
        return DyNDDataDescriptor(nd.asarray(h5arr[key], access='readonly'))

    def __setitem__(self, key, value):
        # HDF5 arrays can be updated
        self.h5arr[key] = value

    def __iter__(self):
        return hdf5_descriptor_iter(self.h5arr)

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
        self.h5arr.append(values_arr.reshape(shape_vals))
        self.h5arr.flush()

