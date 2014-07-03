from __future__ import absolute_import, division, print_function

import numpy as np
from itertools import chain
import h5py
from dynd import nd
import datashape
from datashape import var, dshape
from toolz.curried import pipe, concat, map, partial

from ..dispatch import dispatch
from .core import DataDescriptor
from ..utils import partition_all, get
from ..compatibility import _strtypes, unicode

h5py_attributes = ['chunks', 'compression', 'compression_opts', 'dtype',
                   'fillvalue', 'fletcher32', 'maxshape', 'shape']

__all__ = ['HDF5', 'discover']


@dispatch(h5py.Dataset)
def discover(d):
    s = str(datashape.from_numpy(d.shape, d.dtype))
    return dshape(s.replace('object', 'string'))


def varlen_dtype(dt):
    """ Inject variable length string element for 'O' """
    if "'O'" not in str(dt):
        return dt
    varlen = h5py.special_dtype(vlen=unicode)
    return np.dtype(eval(str(dt).replace("'O'", 'varlen')))



class HDF5(DataDescriptor):
    """
    A Blaze data descriptor which exposes an HDF5 file.

    Parameters
    ----------
    path: string
        Location of hdf5 file on disk
    datapath: string
        Location of array dataset in hdf5
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

    def __init__(self, path, datapath,
                 schema=None, dshape=None, **kwargs):
        self.path = path
        self.datapath = datapath

        if isinstance(schema, _strtypes):
            schema = datashape.dshape(schema)
        if isinstance(dshape, _strtypes):
            dshape = datashape.dshape(dshape)
        if schema and not dshape:
            dshape = var * datashape.dshape(schema)

        if not dshape:
            with h5py.File(path, 'r') as f:
                dset = f.get(datapath)
                if dset:
                    dshape = discover(dset)
                else:
                    raise ValueError("No datashape given or found. "
                             "Please specify dshape or schema keyword args")


        # TODO: provide sane defaults for kwargs
        # Notably chunks and maxshape
        shape = dshape.shape
        dtype = varlen_dtype(dshape[-1].to_numpy_dtype())
        if shape[0] == datashape.Var():
            kwargs['chunks'] = True
            kwargs['maxshape'] = kwargs.get('maxshape', (None,) + shape[1:])
            shape = (0,) + tuple(map(int, shape[1:]))


        with h5py.File(path) as f:
            dset = f.get(datapath)
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
                key = rec_key + key[:-1]
            else:
                key = (rec_key,) + key[:-1]
        with h5py.File(self.path, mode='r') as f:
            arr = f[self.datapath]
            result = np.asarray(arr.__getitem__(key))
        return nd.asarray(result, access='readonly')

    def _get_py(self, key):
        if (isinstance(key, tuple) and
            len(key) > len(self.dshape.shape) and
            isinstance(self.dshape[-1], datashape.Record)):
            rec_key = get(key[-1], self.dshape[-1].names)
            if isinstance(rec_key, tuple):
                key = rec_key + key[:-1]
            else:
                key = (rec_key,) + key[:-1]
        with h5py.File(self.path, mode='r') as f:
            arr = f[self.datapath]
            result = np.asarray(arr.__getitem__(key))
        return result.tolist()

    def __setitem__(self, key, value):
        with h5py.File(self.path) as f:
            arr = f[self.datapath]
            arr[key] = value
        return self

    def _chunks(self, blen=None):
        with h5py.File(self.path, mode='r') as f:
            arr = f[self.datapath]
            if not blen and arr.chunks:
                blen = arr.chunks[0] * 4
            blen = blen or 1024
            for i in range(0, arr.shape[0], blen):
                yield np.array(arr[i:i+blen])

    def __iter__(self):
        return pipe(self.chunks(), map(partial(nd.as_py, tuple=True)), concat)

    def as_dynd(self):
        return self.dynd[:]

    def _extend_chunks(self, chunks):
        with h5py.File(self.path, mode='a') as f:
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
        chunks = partition_all(100, seq)

        with h5py.File(self.path, mode='a') as f:
            dset = f[self.datapath]
            dtype = dset.dtype
            shape = dset.shape
            for chunk in chunks:
                arr = np.asarray(list(chunk), dtype=dtype)
                shape = list(dset.shape)
                shape[0] += len(arr)
                dset.resize(shape)
                dset[-len(arr):] = arr

