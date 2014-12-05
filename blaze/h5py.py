from __future__ import absolute_import, division, print_function

import numpy as np
from toolz import keyfilter, partition_all
from collections import Iterator
import h5py

import datashape
from datashape import DataShape, Record, Mono, dshape, to_numpy, to_numpy_dtype
from datashape.predicates import isrecord, iscollection

from .dispatch import dispatch
from .resource import resource
from .compatibility import _strtypes


h5py_attributes = ['chunks', 'compression', 'compression_opts', 'dtype',
                   'fillvalue', 'fletcher32', 'maxshape', 'shape']


@dispatch((h5py.Group, h5py.File))
def discover(g):
    return DataShape(Record([[k, discover(v)] for k, v in g.items()]))


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


def dataset_from_dshape(file, datapath, ds, **kwargs):
    dtype = varlen_dtype(to_numpy_dtype(ds))
    if datashape.var not in list(ds):
        shape = to_numpy(ds)[0]
    elif len(ds.shape) == 1:
        shape = (0,)
    else:
        raise ValueError("Don't know how to handle varlen nd shapes")

    if shape:
        kwargs['chunks'] = kwargs.get('chunks', True)
        kwargs['maxshape'] = kwargs.get('maxshape', (None,) + shape[1:])

    kwargs2 = keyfilter(h5py_attributes.__contains__, kwargs)
    return file.require_dataset(datapath, shape=shape, dtype=dtype, **kwargs2)

@dispatch(object, (Mono,) + _strtypes)
def create_from_datashape(o, ds, **kwargs):
    return create_from_datashape(o, dshape(ds), **kwargs)

@dispatch((h5py.File, h5py.Group), DataShape)
def create_from_datashape(group, ds, name=None, **kwargs):
    if isinstance(group, type):
        group = h5py.File(kwargs['path'])
    assert isrecord(ds)
    for name, sub_ds in ds[0].dict.items():
        if isrecord(sub_ds):
            g = group.require_group(name)
            create_from_datashape(g, sub_ds, **kwargs)
        else:
            dataset_from_dshape(file=group.file,
                                datapath='/'.join([group.name, name]),
                                ds=sub_ds, **kwargs)


def hdf5_from_datashape(target, ds, **kwargs):
    if isinstance(ds, _strtypes):
        ds = dshape(ds)
    if isinstance(target, _strtypes):
        if '::' in target:
            path, datapath = target.split('::')
        else:
            path, datapath = target, ''
        while datapath:
            datapath, name = datapath.rsplit('/', 1)
            ds = Record([[name, ds]])
        ds = dshape(ds)
        target = h5py.File(path)
    create_from_datashape(target, ds, **kwargs)
    return target


@dispatch(h5py.Dataset, np.ndarray)
def into(dset, x, **kwargs):
    assert not isinstance(dset, type)
    shape = list(dset.shape)
    shape[0] += len(x)
    dset.resize(shape)
    dset[-len(x):] = x
    return dset


@dispatch(h5py.Dataset, (list, tuple, set, Iterator))
def into(dset, seq, chunksize=int(2**10), **kwargs):
    assert not isinstance(dset, type)
    for chunk in partition_all(chunksize, seq):
        into(dset, into(np.ndarray, chunk, dshape=discover(dset).measure), **kwargs)
    return dset


@dispatch((h5py.Group, h5py.Dataset))
def drop(h):
    del h.file[h.name]


@dispatch(h5py.File)
def drop(h):
    os.remove(h.filename)


@resource.register('.+\.hdf5')
def resource_h5py_file(uri, datapath=None, **kwargs):
    f = h5py.File(uri)
    ds = kwargs.pop('dshape', None)
    if ds:
        if datapath and datapath != '/':
            uri = uri + '::' + datapath.rstrip('/')
        f.close()
        f = hdf5_from_datashape(uri, ds, **kwargs)
    if datapath:
        return f[datapath]
    else:
        return f
