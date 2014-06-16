from __future__ import absolute_import, division, print_function

from itertools import chain
from dynd import nd
import datashape
from datashape.internal_utils import IndexCallable

from .utils import validate, coerce, coerce_to_ordered, ordered_index
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

        def dtype_of(chunk):
            return str(len(chunk) * self.schema)

        self._extend_chunks((nd.array(chunk, dtype=dtype_of(chunk))
                             for chunk in chunks))

    def _extend_chunks(self, chunks):
        self.extend((row for chunk in chunks
                         for row in nd.as_py(chunk, tuple=True)))

    def chunks(self, **kwargs):
        def dshape(chunk):
            return str(len(chunk) * self.dshape.subshape[0])

        chunks = list(self._chunks(**kwargs))
        return (nd.array(chunk, dtype=dshape(chunk)) for chunk in chunks)

    def _chunks(self, blen=100):
        return partition_all(blen, iter(self))

    def as_dynd(self):
        return self.dynd[:]

    def as_py(self):
        if isdimension(self.dshape[0]):
            return tuple(self)
        else:
            return tuple(nd.as_py(self.as_dynd(), tuple=True))

    def __array__(self):
        return nd.as_numpy(self.as_dynd())

    @property
    def py(self):
        return IndexCallable(self.get_py)

    @property
    def dynd(self):
        return IndexCallable(self.get_dynd)

    def get_py(self, key):
        key = ordered_index(key, self.dshape)
        subshape = self.dshape._subshape(key)
        if hasattr(self, '_get_py'):
            result = self._get_py(key)
        elif hasattr(self, '_get_dynd'):
            result = self._get_dynd(key)
        else:
            raise AttributeError("Data Descriptor defines neither "
                                 "_get_py nor _get_dynd.  Can not index")
        return coerce(subshape, result)

    def get_dynd(self, key):
        key = ordered_index(key, self.dshape)
        subshape = self.dshape._subshape(key)
        if hasattr(self, '_get_dynd'):
            result = self._get_dynd(key)
        elif hasattr(self, '_get_py'):
            result = nd.array(self._get_py(key), type=str(subshape))
        else:
            raise AttributeError("Data Descriptor defines neither "
                                 "_get_py nor _get_dynd.  Can not index")
        return nd.array(result, type=str(subshape))

    def __iter__(self):
        if not isdimension(self.dshape[0]):
            raise TypeError("Data Descriptor not iterable, has dshape %s" %
                            self.dshape)
        schema = self.dshape.subshape[0]
        try:
            seq = self._iter()
        except NotImplementedError:
            seq = iter(nd.as_py(self.as_dynd(), tuple=True))
        if not isdimension(self.dshape[0]):
            yield coerce(self.dshape, nd.as_py(self.as_dynd(), tuple=True))
        else:
            for block in partition_all(100, seq):
                x = coerce(len(block) * schema, block)
                for row in x:
                    yield row

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

    @property
    def columns(self):
        rec = self.schema[0]
        if isinstance(rec, datashape.Record):
            return rec.names
        else:
            raise TypeError('Columns attribute only valid on tabular '
                            'datashapes of records, got %s' % self.dshape)


from ..dispatch import dispatch
from blaze.expr.table import Join, TableExpr
from blaze.expr.core import Expr
@dispatch((Join, Expr), DataDescriptor)
def compute(t, ddesc):
    return compute(t, iter(ddesc))  # use Python streaming by default
