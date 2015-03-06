from __future__ import absolute_import, division, print_function

import os

import numpy as np

from toolz import first
from .dispatch import dispatch

import datashape

import shutil
from blaze.utils import tmpfile
from odo import resource

try:
    import tables as tb
    from tables import Table
except ImportError:
    Table = type(None)

__all__ = ['PyTables']


def dtype_to_pytables(dtype):
    """ Convert NumPy dtype to PyTable descriptor

    Examples
    --------
    >>> from tables import Int32Col, StringCol, Time64Col
    >>> dt = np.dtype([('name', 'S7'), ('amount', 'i4'), ('time', 'M8[us]')])
    >>> dtype_to_pytables(dt)  # doctest: +SKIP
    {'amount': Int32Col(shape=(), dflt=0, pos=1),
     'name': StringCol(itemsize=7, shape=(), dflt='', pos=0),
     'time': Time64Col(shape=(), dflt=0.0, pos=2)}
    """
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


def PyTables(path, datapath, dshape=None, **kwargs):
    """Create or open a ``tables.Table`` object.

    Parameters
    ----------
    path : str
        Path to a PyTables HDF5 file.
    datapath : str
        The name of the node in the ``tables.File``.
    dshape : str or datashape.DataShape
        DataShape to use to create the ``Table``.

    Returns
    -------
    t : tables.Table

    Examples
    --------
    >>> from blaze.utils import tmpfile
    >>> # create from scratch
    >>> with tmpfile('.h5') as f:
    ...     t = PyTables(filename, '/bar',
    ...                  dshape='var * {volume: float64, planet: string[10, "A"]}')
    ...     data = [(100.3, 'mars'), (100.42, 'jupyter')]
    ...     t.append(data)
    ...     t[:]  # doctest: +SKIP
    ...
    array([(100.3, b'mars'), (100.42, b'jupyter')],
          dtype=[('volume', '<f8'), ('planet', 'S10')])
    """
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

    if dshape:
        if isinstance(dshape, str):
            dshape = datashape.dshape(dshape)
        if dshape[0] == datashape.var:
            dshape = dshape.subshape[0]
        dtype = dtype_to_pytables(datashape.to_numpy_dtype(dshape))
    else:
        dtype = None

    if os.path.exists(path):
        possibly_create_table(path, dtype)
    else:
        with tmpfile('.h5') as filename:
            possibly_create_table(filename, dtype)
            shutil.copyfile(filename, path)
    return tb.open_file(path, mode='a').get_node(datapath)


@dispatch(Table)
def chunks(b, chunksize=2**15):
    start = 0
    n = len(b)
    while start < n:
        yield b[start:start + chunksize]
        start += chunksize


@dispatch(Table, int)
def get_chunk(b, i, chunksize=2**15):
    start = chunksize * i
    stop = chunksize * (i + 1)
    return b[start:stop]


@resource.register('.+\.h5')
def resource_pytables(path, datapath=None, **kwargs):
    if not datapath:
        return tb.open_file(path)
    else:
        return PyTables(path, datapath, **kwargs)
