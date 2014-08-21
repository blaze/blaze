from __future__ import absolute_import, division, print_function
from multipledispatch import dispatch
from .hdf5 import HDF5
from .sql import SQL
import tables as tb


@dispatch(HDF5)
def drop(h):
    pass


@dispatch(tb.Table)
def drop(t):
    pass


@dispatch(SQL)
def drop(s):
    s.table.drop(s.engine)


@dispatch(CSV)
def drop(c):
    c.remove()
