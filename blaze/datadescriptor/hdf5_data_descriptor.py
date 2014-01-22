from __future__ import absolute_import, division, print_function

import numpy as np
from dynd import nd
import datashape

from . import IDataDescriptor, Capabilities
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

    def _open(self):
        self.f = tb.open_file(self.filename)
        return self.f.get_node(self.f.root, self.datapath)

    def _close(self):
        self.f.close()

    def __init__(self, filename, datapath):
        self.filename = filename
        self.datapath = datapath
        obj = self._open()
        # We are going to support both homogeneous and heterogeneous
        # datasets, but not VL types (VLArray) for the time being.
        if not isinstance(obj, (tb.Array, tb.Table)):
            raise TypeError(('object is not a supported HDF5 dataset, '
                             'it has type %r') % type(obj))
        self._close()

    @property
    def dshape(self):
        # This cannot be cached because the Array can change the dshape
        h5arr = self._open()
        odshape = datashape.from_numpy(h5arr.shape, h5arr.dtype)
        self._close()
        return odshape

    @property
    def capabilities(self):
        """The capabilities for the HDF5 arrays."""
        h5arr = self._open()
        caps = Capabilities(
            # HDF5 arrays can be updated
            immutable = False,
            # HDF5 arrays are concrete
            deferred = False,
            # HDF5 arrays are persistent
            persistent = True,
            # HDF5 arrays can be appended efficiently for EArrays and Tables
            appendable = isinstance(h5arr, (tb.EArray, tb.Table)),
            remote = False,
            )
        self._close()
        return caps

    def dynd_arr(self):
        # Positionate at the beginning of the file
        h5arr = self._open()
        h5arr = nd.array(h5arr[:], dtype=h5arr.dtype)
        self._close()
        return h5arr

    def __array__(self):
        h5arr = self._open()
        h5arr = h5arr[:]
        self._close()
        return h5arr

    def __len__(self):
        h5arr = self._open()
        arrlen = len(h5arr)
        self._close()
        return arrlen

    def __getitem__(self, key):
        h5arr = self._open()
        h5arr = h5arr[key]
        self._close()
        # The returned arrays are temporary buffers,
        # so must be flagged as readonly.
        return DyNDDataDescriptor(nd.asarray(h5arr, access='readonly'))

    def __setitem__(self, key, value):
        # HDF5 arrays can be updated
        h5arr = self._open()
        h5arr[key] = value
        self._close()

    def __iter__(self):
        h5arr = self._open()
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
        h5arr = self._open()
        h5arr.append(values_arr.reshape(shape_vals))
        h5arr.flush()
        self._close()
