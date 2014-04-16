from __future__ import absolute_import, division, print_function

from itertools import chain
from dynd import nd
import datashape

from .utils import validate, coerce
from ..utils import partition_all

__all__ = ['DataDescriptor', 'copy']


class DataDescriptor(object):
    """
    The Blaze data descriptor is an interface which exposes
    data to Blaze. The data descriptor doesn't implement math
    or any other kind of functions, its sole purpose is providing
    single and multi-dimensional data to Blaze via a data shape,
    and the indexing/iteration interfaces.

    Indexing and python iteration must always return data descriptors,
    this is the python interface to the data. A summary of the
    data access patterns for a data descriptor dd, in the
    0.3 version of blaze are:

     - descriptor integer indexing
            child_dd = dd[i, j, k]
     - slice indexing
            child_dd = dd[i:j]
     - descriptor outer/leading dimension iteration
            for child_dd in dd: do_something(child_dd)
     - memory access via dynd array (either using dynd library
       to process, or directly depending on the ABI of the dynd
       array object, which will be stabilized prior to dynd 1.0)

    The descriptor-based indexing methods operate only through the
    Python interface, JIT-compiled access should be done through
    processing the dynd type and corresponding array metadata.
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
