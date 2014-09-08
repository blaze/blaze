import os

import numpy as np
import tables as tb

from cytoolz import compose, first

import datashape

import shutil
from blaze.utils import tmpfile


def sort_dtype(items, names):
    return np.dtype(sorted(items, key=compose(names.index, first)))


def to_tables_descr(dtype):
    d = {}
    for (pos, name), (dtype, _) in zip(enumerate(dtype.names),
                                       map(dtype.fields.__getitem__,
                                           dtype.names)):
        if issubclass(dtype.type, np.datetime64):
            tdtype = tb.Description({name: tb.Time64Col(pos=pos)}),
        else:
            tdtype = tb.descr_from_dtype(np.dtype([(name, dtype)]))
        el = first(tdtype)
        getattr(el, name)._v_pos = pos
        d.update(el._v_colobjects)
    return d


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
        dtype = to_tables_descr(datashape.to_numpy_dtype(dshape))
    else:
        dtype = None

    if os.path.exists(path):
        possibly_create_table(path, dtype)
    else:
        with tmpfile('.h5') as filename:
            possibly_create_table(filename, dtype)
            shutil.copyfile(filename, path)
    return tb.open_file(path).get_node(datapath)
