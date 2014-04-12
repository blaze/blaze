from __future__ import absolute_import, division, print_function

__all__ = ['DDesc', 'copy']

import abc

from blaze.error import StreamingDimensionError
from blaze.compute.strategy import CKERNEL
from .util import validate

from itertools import chain
from dynd import nd


class DDesc(object):
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
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def dshape(self):
        """
        Returns the datashape for the data behind this datadescriptor.
        Every data descriptor implementation must provide a dshape.
        """
        # TODO: Does dshape make sense for a data descriptor? A data descriptor
        # may have a lower-level concept of a data type that corresponds to a
        # higher-level data shape. IMHO dshape should be on Array only
        raise NotImplementedError

    @abc.abstractproperty
    def capabilities(self):
        """A container for the different capabilities."""
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self):
        """
        This returns an iterator/generator which iterates over
        the outermost/leading dimension of the data. If the
        dimension is not also a stream, __len__ should also
        be implemented. The iterator must return data
        descriptors.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, key):
        """
        This does integer/slice indexing, producing another
        data descriptor.
        """
        raise NotImplementedError

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
        from .as_py import ddesc_as_py
        return self.extend((row for chunk in chunks for row in nd.as_py(chunk)))

    def iterchunks(self, **kwargs):
        def dshape(chunk):
            n = len(chunk)
            s = str(self.dshape)
            return str(n) + ' * ' + ' * '.join(s.split(' * ')[1:])

        chunks = self._iterchunks(**kwargs)
        return (nd.array(chunk, dtype=dshape(chunk)) for chunk in chunks)

    def _iterchunks(self, blen=100):
        raise NotImplementedError()

    def getattr(self, name):
        raise NotImplementedError('this data descriptor does not support attribute access')

    def dynd_arr(self):
        """Concrete data descriptors must provide their array data
           as a dynd array, accessible via this method.
        """
        if not self.capabilities['deferred']:
            raise NotImplementedError((
                'Data descriptor of type %s claims '
                'claims to not being deferred, but did not '
                'override dynd_arr()') % type(self))
        else:
            raise TypeError((
                'Data descriptor of type %s is deferred') % type(self))

def copy(src, dest, **kwargs):
    """ Copy content from one data descriptor to another """
    dest.extend_chunks(src.iterchunks(**kwargs))
