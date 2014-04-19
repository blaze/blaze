from __future__ import absolute_import, division, print_function

from itertools import chain
from dynd import nd
import datashape

from .utils import validate, coerce
from ..utils import partition_all

__all__ = ['DataDescriptor']


def isdimension(ds):
    return isinstance(ds, (datashape.Var, datashape.Fixed))


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
    as_dynd - load entire dataset into memory as a DyND array
    """

    def extend(self, rows):
        """ Extend data with many rows
        """
        if not self.appendable or self.immutable:
            raise TypeError('Data Descriptor not appendable')
        rows = iter(rows)
        row = next(rows)
        if not validate(self.schema, row):
            raise ValueError('Invalid data:\n\t %s \nfor dshape \n\t%s' %
                    (str(row), self.schema))
        self._extend(chain([row], rows))


    def extend_chunks(self, chunks):
        if not self.appendable or self.immutable:
            raise TypeError('Data Descriptor not appendable')
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

    def as_dynd(self):
        return nd.array(self, dtype=str(self.dshape))

    def as_py(self):
        if isdimension(self.dshape[0]):
            return list(self)
        else:
            return self.as_dynd()

    def __array__(self):
        return nd.as_numpy(self.as_dynd())

    def __getitem__(self, key):
        if hasattr(self, '_getitem'):
            return coerce(self.schema, self._getitem(key))
        else:
            return self.as_dynd()[key]

    def __iter__(self):
        try:
            for row in self._iter():
                yield coerce(self.schema, row)
        except NotImplementedError:
            py = nd.as_py(self.as_dynd())
            if isdimension(self.dshape[0]):
                for row in py:
                    yield row
            else:
                yield py

    def _iter(self):
        raise NotImplementedError()

    _dshape = None
    @property
    def dshape(self):
        return datashape.dshape(self._dshape or datashape.Var() * self.schema)

    _schema = None
    @property
    def schema(self):
        if self._schema:
            return datashape.dshape(self._schema)
        if isdimension(self.dshape[0]):
            return self.dshape.subarray(1)
        raise TypeError('Datashape is not indexable to schema\n%s' %
                        self.dshape)
