from .data_descriptor import DDesc
from .dynd_data_descriptor import DyND_DDesc
from ..utils import partition_all
import h5py
import datashape
from dynd import nd
import numpy as np

h5py_attributes = ['chunks', 'compression', 'compression_opts', 'dtype',
                   'fillvalue', 'fletcher32', 'maxshape', 'shape']

class HDF5_DDesc(DDesc):
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
    **kwargs:
        Options to send to h5py - see h5py.File.create_dataset for options
    """

    def __init__(self, path, datapath, mode='r', dshape=None, **kwargs):
        self.path = path
        self.datapath = datapath
        self.mode = mode


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
                    print(datapath, shape, dtype)
                    f.create_dataset(datapath, shape, dtype=dtype, **kwargs)
            else:
                # TODO: test provided dshape against given dshape
                dshape2 = datashape.from_numpy(dset.shape, dset.dtype)
                if dshape and dshape != dshape2:
                    raise ValueError('Inconsistent datashapes.'
                            'Given: %s\nFound: %s' % (dshape, dshape2))
                dshape = dshape2

        attributes = self.attributes()
        if attributes['chunks']:
            # is there a better way to do this?
            words = str(dshape).split(' * ')
            dshape = 'var * ' + ' * '.join(words[1:])
            dshape = datashape.dshape(dshape)

        self._dshape = dshape

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

    def iterchunks(self, blen=100):
        with h5py.File(self.path, mode='r') as f:
            arr = f[self.datapath]
            for i in range(0, arr.shape[0], blen):
                yield nd.asarray(np.array(arr[i:i+blen]), access='readonly')

    def __iter__(self):
        pass

    @property
    def dshape(self):
        return self._dshape

    @property
    def capabilities(self):
        return {'immutable': False,
                'deferred': False,
                'persistent': True,
                'appendable': True,
                'remote': False}

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
