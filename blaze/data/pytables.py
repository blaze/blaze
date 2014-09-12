import os

import numpy as np
import tables as tb

from toolz import first
from ..dispatch import dispatch

import datashape as ds

import shutil
from blaze.utils import tmpfile


@dispatch(tb.Description, np.dtype)
def into(_, dtype):
    d = {}
    for pos, name in enumerate(dtype.names):
        dt, _ = dtype.fields[name]
        if issubclass(dt.type, np.datetime64):
            tdtype = tb.Description({name: tb.Time64Col(pos=pos)}),
        else:
            tdtype = tb.descr_from_dtype(np.dtype([(name, dt)]))
        el = first(tdtype)
        getattr(el, name)._v_pos = pos
        d.update(el._v_colobjects)
    return d


@dispatch(np.dtype, tb.Description)
def into(a, b):
    pass


@dispatch(np.dtype, ds.DataShape)
def into(_, b):
    return ds.to_numpy_dtype(b)


@dispatch(ds.DataShape, np.dtype)
def into(a, b):
    pass


def PyTables(path, datapath, dshape=None):
    def possibly_create_table(filename, dtype):
        f = tb.open_file(filename, mode='a')
        try:
            if datapath not in f:
                if dtype is None:
                    raise ValueError('dshape cannot be None and datapath not'
                                     ' in file')
                else:
                    f.create_table('/', datapath.lstrip('/'), description=dtype)
        finally:
            f.close()

    if dshape is not None:
        dtype = into(tb.Description, into(np.dtype, ds.dshape(dshape)))
    else:
        dtype = None

    if os.path.exists(path):
        possibly_create_table(path, dtype)
    else:
        with tmpfile('.h5') as filename:
            possibly_create_table(filename, dtype)
            shutil.copyfile(filename, path)
    return tb.open_file(path, mode='a').get_node(datapath)
