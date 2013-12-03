from __future__ import absolute_import

__all__ = ['IDataDescriptor']

import abc
import ctypes
import contextlib

from blaze.error import StreamingDimensionError
from blaze.compute.strategy import current_strategy

class IDataDescriptor:
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

    deferred = False

    @property
    def persistent(self):
        return False

    @abc.abstractproperty
    def is_concrete(self):
        """Returns True if the data can be returned as an
           in-memory dynd array. Returns False for deferred
           expressions and persistent arrays.
        """
        raise NotImplementedError

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
    def writable(self):
        """
        Returns True if the data is writable,
        False otherwise.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def immutable(self):
        """
        Returns True if the data is immutable,
        False otherwise.
        """
        raise NotImplementedError

    #@abc.abstractproperty   # XXX should be there
    def appendable(self):
        """
        Returns True if the data is appendable,
        False otherwise.
        """
        raise NotImplementedError

    def __len__(self):
        """
        The default implementation of __len__ is for the
        behavior of a streaming dimension, where the size
        of the dimension isn't known ahead of time.

        If a data descriptor knows its dimension size,
        it should implement __len__, and provide the size
        as an integer.
        """
        raise StreamingDimensionError('Cannot get the length of'
                        ' a streaming dimension')

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

    #@abc.abstractmethod   # XXX should be there
    def append(self, values):
        """
        This allows appending values in the data descriptor.
        """
        return NotImplementedError

    @property
    def strategy(self):
        return current_strategy()

    def dynd_arr(self):
        """Concrete data descriptors must provide their array data
           as a dynd array, accessible via this method.
        """
        if self.is_concrete:
            raise NotImplementedError(('Data descriptor of type %s' +
                        ' claims to be concrete, but did not'
                        ' override dynd_arr()') % type(self))
        else:
            raise TypeError(('Data descriptor of type %s is not ' +
                        'concrete') % type(self))

