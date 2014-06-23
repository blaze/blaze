from __future__ import absolute_import, division, print_function

import numpy as np
from itertools import chain
import h5py
from dynd import nd
import datashape
from datashape import var

from ..dispatch import dispatch
from .core import DataDescriptor
from ..utils import partition_all, get
from ..compatibility import _strtypes

h5py_attributes = ['chunks', 'compression', 'compression_opts', 'dtype',
                   'fillvalue', 'fletcher32', 'maxshape', 'shape']

__all__ = ['HDF5', 'discover']


@dispatch(h5py.Dataset)
def discover(d):
    return datashape.from_numpy(d.shape, d.dtype)


class HDF5(DataDescriptor):
    """
    A Blaze data descriptor which exposes an HDF5 file.

    Parameters
    ----------
    path: string
        Location of hdf5 file on disk
    datapath: string
        Location of array dataset in hdf5
    mode : string
        r, w, rw+
    dshape: string or Datashape
        a datashape describing the data
    schema: string or DataShape
        datashape describing one row of data
    **kwargs:
        Options to send to h5py - see h5py.File.create_dataset for options
    """
    immutable = False
    deferred = False
    persistent = True
    appendable = True
    remote = False

    def __init__(self, path, datapath, mode='r',
                 schema=None, dshape=None, **kwargs):
        self.path = path
        self.datapath = datapath
        self.mode = mode

        if isinstance(schema, _strtypes):
            schema = datashape.dshape(schema)
        if isinstance(dshape, _strtypes):
            dshape = datashape.dshape(dshape)
        if schema and not dshape:
            dshape = var * datashape.dshape(schema)

        # TODO: provide sane defaults for kwargs
        # Notably chunks and maxshape
        if dshape:
            dshape = datashape.dshape(dshape)
            shape = dshape.shape
            dtype = dshape[-1].to_numpy_dtype()
            if shape[0] == datashape.Var():
                kwargs['chunks'] = True
                kwargs['maxshape'] = kwargs.get('maxshape', (None,) + shape[1:])
                shape = (0,) + tuple(map(int, shape[1:]))

        with h5py.File(path, mode) as f:
            dset = f.get(datapath)
            if dset:
                file_dshape = discover(dset)
                if dshape and file_dshape != dshape:
                    raise TypeError("Inconsistent dshapes given:\n"
                                    "\tGiven: %s\n"
                                    "\tFound: %s\n" % (dshape, file_dshape))
                else:
                    dshape = file_dshape
            if not dset:
                f.create_dataset(datapath, shape, dtype=dtype, **kwargs)

        attributes = self.attributes()
        if attributes['chunks']:
            dshape = var * dshape.subshape[0]

        self._dshape = dshape
        self._schema = schema

    def attributes(self):
        with h5py.File(self.path, 'r') as f:
            arr = f[self.datapath]
            result = dict((attr, getattr(arr, attr))
                          for attr in h5py_attributes)
        return result

    def _get_dynd(self, key):
        if (isinstance(key, tuple) and
            len(key) > len(self.dshape.shape) and
            isinstance(self.dshape[-1], datashape.Record)):
            rec_key = get(key[-1], self.dshape[-1].names)
            if isinstance(rec_key, tuple):
                rec_key = list(rec_key)
            key = (rec_key,) + key[:-1]
            print(key)
        with h5py.File(self.path, mode='r') as f:
            arr = f[self.datapath]
            result = np.asarray(arr.__getitem__(key))
        return nd.asarray(result, access='readonly')

    def __setitem__(self, key, value):
        with h5py.File(self.path, mode=self.mode) as f:
            arr = f[self.datapath]
            arr[key] = value
        return self

    def _chunks(self, blen=100):
        with h5py.File(self.path, mode='r') as f:
            arr = f[self.datapath]
            for i in range(0, arr.shape[0], blen):
                yield np.array(arr[i:i+blen])

    def as_dynd(self):
        return self.dynd[:]

    def _extend_chunks(self, chunks):
        if 'w' not in self.mode and 'a' not in self.mode:
            raise ValueError('Read only')

        with h5py.File(self.path, mode=self.mode) as f:
            dset = f[self.datapath]
            dtype = dset.dtype
            shape = dset.shape
            for chunk in chunks:
                arr = nd.as_numpy(chunk, allow_copy=True)
                shape = list(dset.shape)
                shape[0] += len(arr)
                dset.resize(shape)
                dset[-len(arr):] = arr

    def _extend(self, seq):
        self.extend_chunks(partition_all(100, seq))

    def _iter(self):
        return chain.from_iterable(self.chunks())
