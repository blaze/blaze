"""
This file defines the Table class.

A Table is constructed from Array objects which are columns.  Hence the
data layout is columnar, and columns can be added and removed
efficiently.  It is also meant to provide easy filtering based on column
conditions.
"""

from __future__ import absolute_import, division, print_function

import datashape
from .array import Array


class Table(object):
    """
    Table(cols, labels=None, **kwargs)

    Create a new Table from `cols` with optional `names`.

    Parameters
    ----------
    columns : tuple or list of column objects
        The list of column data to build the Table object.  This list would
        typically be made of Blaze Array objects, but could also understand
        DyND or NumPy arrays.  A list of lists or tuples is valid too, as
        long as they can be converted into barray objects.
    labels : list of strings
        The list of names for the columns.  The names in this list have
        to be specified in the same order as the `cols`.  If not passed, the
        names will be chosen as 'f0' for the first column, 'f1' for the
        second and so on so forth (NumPy convention).
    kwargs : list of parameters or dictionary
        Allows to pass additional arguments supported by Array
        constructors in case new columns need to be built.

    Notes
    -----
    Columns passed as Array objects are not be copied, so their settings
    will stay the same, even if you pass additional arguments.

    """
    def __init__(self, columns, labels=None, **kwargs):
        arr_cols = {}
        for i, column in enumerate(columns):
            if isinstance(column, Array):
                arr_cols['f%d'%i] = column
            else:
                try:
                    arr_cols['f%d'%i] = array(column)
                except:
                    raise TypeError(
                        ('Constructing a blaze table directly '
                         'requires columns that can be converted '
                         'to Blaze arrays') % (type(data)))
        self._cols = arr_cols
        self.labels = labels or arr_cols.keys()


    @property
    def dshape(self):
        # Build a dshape out of the columns and labels
        return XXX

    @property
    def deferred(self):
        return XXX

    def __array__(self):
        import numpy as np

        # Expose PEP-3118 buffer interface for columns
        return XXX

    def __iter__(self):
        return XXX

    def __getitem__(self, key):
        return Array(self._data.__getitem__(key))

    def __setitem__(self, key, val):
        self._data.__setitem__(key, val)

    def __len__(self):
        shape = self.dshape.shape
        if shape:
            return shape[0]
        raise IndexError('Scalar blaze arrays have no length')

    def __str__(self):
        if hasattr(self._data, '_printer'):
            return XXX
        return XXX

    def __repr__(self):
        if hasattr(self._data, "_printer_repr"):
            return XXX
        return XXX
