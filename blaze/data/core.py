from __future__ import absolute_import, division, print_function

from itertools import chain
from dynd import nd
import datashape

from .utils import validate, coerce
from ..utils import partition_all

__all__ = ['DataDescriptor', 'copy']


class DataDescriptor(object):
    """
    Standard interface to data storage

    Data descriptors provide read and write access to common data storage
    systems like csv, json, HDF5, and SQL.

    They provide Pythonic iteration over these resources as well as efficient
    chunked access with DyND arrays.

    Data Descriptors implement the following methods:

    __iter__ - iterate over storage, getting results as Python objects
    chunks - iterate over storage, getting results as DyND arrays
    extend - insert new data into storage (if possible.)
             Consumes a sequence of core Python objects
    extend_chunks - insert new data into storage (if possible.)
             Consumes a sequence of DyND arrays
    dynd_arr - load entire dataset into memory as a DyND array
    """
    def append(self, value):
        """
        This allows appending values in the data descriptor.
        """
        self.extend([value])

    def extend(self, rows):
        """ Extend data with many rows

        See Also:
            append
        """
        rows = iter(rows)
        row = next(rows)
        if not validate(self.schema, row):
            raise ValueError('Invalid data:\n\t %s \nfor dshape \n\t%s' %
                    (str(row), self.schema))
        self._extend(chain([row], rows))


    def extend_chunks(self, chunks):
        self._extend_chunks((nd.array(chunk) for chunk in chunks))

    def _extend_chunks(self, chunks):
        self.extend((row for chunk in chunks for row in nd.as_py(chunk)))

    def chunks(self, **kwargs):
        def dshape(chunk):
            return str(len(chunk) * self.dshape.subarray(1))

        chunks = self._chunks(**kwargs)
        return (nd.array(chunk, dtype=dshape(chunk)) for chunk in chunks)

    def _chunks(self, blen=100):
        return partition_all(blen, iter(self))

    def getattr(self, name):
        raise NotImplementedError('this data descriptor does not support attribute access')

    def dynd_arr(self):
        return nd.array(self, dtype=str(self.dshape))

    def __array__(self):
        return nd.as_numpy(self.dynd_arr())

    def __getitem__(self, key):
        if hasattr(self, '_getitem'):
            return coerce(self.schema, self._getitem(key))
        else:
            return self.dynd_arr()[key]

    def __iter__(self):
        for row in self._iter():
            yield coerce(self.schema, row)

    _dshape = None
    @property
    def dshape(self):
        return datashape.dshape(self._dshape or datashape.Var() * self.schema)

    _schema = None
    @property
    def schema(self):
        return datashape.dshape(self._schema or self.dshape.subarray(1))

def copy(src, dest, **kwargs):
    """ Copy content from one data descriptor to another """
    dest.extend_chunks(src.chunks(**kwargs))
