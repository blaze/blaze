import numpy as np
from itertools import chain
import h5py
from dynd import nd
import datashape

from .core import DataDescriptor
from ..utils import partition_all

h5py_attributes = ['chunks', 'compression', 'compression_opts', 'dtype',
                   'fillvalue', 'fletcher32', 'maxshape', 'shape']

__all__ = ['HDF5']

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

    def __init__(self, path, datapath, mode='r', schema=None, dshape=None, **kwargs):
        self.path = path
        self.datapath = datapath
        self.mode = mode

        if schema and not dshape:
            dshape = 'var * ' + str(schema)

        # TODO: provide sane defaults for kwargs
        # Notably chunks and maxshape
        if dshape:
            dshape = datashape.dshape(dshape)
            shape = dshape.shape
            dtype = datashape.to_numpy_dtype(dshape[-1])
            if shape[0] == datashape.Var():
                kwargs['chunks'] = True
                kwargs['maxshape'] = kwargs.get('maxshape', (None,) + shape[1:])
                shape = (0,) + tuple(map(int, shape[1:]))

        with h5py.File(path, mode) as f:
            dset = f.get(datapath)
            if dset is None:
                if dshape is None:
                    raise ValueError('No dataset or dshape provided')
                else:
                    f.create_dataset(datapath, shape, dtype=dtype, **kwargs)
            else:
                dshape2 = datashape.from_numpy(dset.shape, dset.dtype)
                dshape = dshape2
                # TODO: test provided dshape against given dshape
                # if dshape and dshape != dshape2:
                #     raise ValueError('Inconsistent datashapes.'
                #             '\nGiven: %s\nFound: %s' % (dshape, dshape2))

        attributes = self.attributes()
        if attributes['chunks']:
            # is there a better way to do this?
            words = str(dshape).split(' * ')
            dshape = 'var * ' + ' * '.join(words[1:])
            dshape = datashape.dshape(dshape)

        self._dshape = dshape
        self._schema = schema

    def attributes(self):
        with h5py.File(self.path, 'r') as f:
            arr = f[self.datapath]
            result = dict((attr, getattr(arr, attr))
                            for attr in h5py_attributes)
        return result

    def __getitem__(self, key):
        with h5py.File(self.path, mode='r') as f:
            arr = f[self.datapath]
            result = np.asarray(arr[key])
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

    def dynd_arr(self):
        return self[:]

    def _extend_chunks(self, chunks):
        if 'w' not in self.mode and 'a' not in self.mode:
            raise ValueError('Read only')

        with h5py.File(self.path, mode=self.mode) as f:
            dset = f[self.datapath]
            dtype = dset.dtype
            shape = dset.shape
            for chunk in chunks:
                arr = np.array(chunk, dtype=dtype)
                shape = list(dset.shape)
                shape[0] += len(arr)
                dset.resize(shape)
                dset[-len(arr):] = arr

    def _extend(self, seq):
        self.extend_chunks(partition_all(100, seq))

    def __iter__(self):
        return chain.from_iterable(self.chunks())
