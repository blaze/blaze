"""
Wrappers around chunked arrays ( carray ) supporting both
in-memory and on-disk storage.
"""

from blaze import carray

from blaze.sources.descriptors.byteprovider import ByteProvider
from blaze.byteproto import CONTIGUOUS, CHUNKED, STREAM, ACCESS_ALLOC
from blaze.datadescriptor import CArrayDataDescriptor
from blaze.datashape.coretypes import dynamic, from_numpy, to_numpy
from blaze.params import params, to_cparams
from blaze.layouts.scalar import ChunkedL

import numpy as np

#------------------------------------------------------------------------
# Chunked Array
#------------------------------------------------------------------------

class CArraySource(ByteProvider):
    """ Chunked array is the default storage engine for Blaze arrays
    when no layout is specified. """

    read_capabilities  = CHUNKED
    write_capabilities = CHUNKED
    access_capabilities = ACCESS_ALLOC

    def __init__(self, data=None, dshape=None, params=None):
        """ CArray object passed directly into the constructor,
        ostensibly this is just a thin wrapper that consumes a
        reference.
        """
        # need at least one of the three
        assert (data is not None) or (dshape is not None) or \
               (params.get('storage'))

        # TODO: clean up ugly conditionals

        if params:
            cparams, rootdir, format_flavor = to_cparams(params)
        else:
            rootdir = None
            cparams = None

        if dshape:
            dtype = to_numpy(dshape)
            self.ca = carray.carray(data, dtype, rootdir=rootdir)
        else:
            self.ca = carray.carray(data, rootdir=rootdir, cparams=cparams)

    @classmethod
    def empty(self, dshape):
        """ Create a CArraySource from a datashape specification,
        downcasts into Numpy dtype and shape tuples if possible
        otherwise raises an exception.
        """
        shape, dtype = from_numpy(dshape)
        return CArraySource(carray([], dtype))

    # Get a READ descriptor the source
    def read_desc(self):
        return CArrayDataDescriptor('carray_dd', self.ca.nbytes, self.ca)

    # Return the layout of the dataa
    def default_layout(self):
        return ChunkedL(self, cdimension=0)

    @property
    def nchunks(self):
        """ Number of chunks """
        return self.ca.nchunks

    @property
    def partitions(self):
        """
        Return the partitions of elemenets in the array. The data
        bounds of each chunk.
        """
        return self.ca.partitions

    @staticmethod
    def infer_datashape(source):
        """
        The user has only provided us with a Python object ( could be
        a buffer interface, a string, a list, list of lists, etc) try
        our best to infer what the datashape should be in the context of
        what it would mean as a CArray.
        """
        if isinstance(source, np.ndarray):
            return from_numpy(source.shape, source.dtype)
        elif isinstance(source, CArraySource):
            return from_numpy(source.ca.shape, source.ca.dtype)
        elif isinstance(source, list):
            # TODO: um yeah, we'd don't actually want to do this
            cast = np.array(source)
            return from_numpy(cast.shape, cast.dtype)
        else:
            return dynamic

    # Structural Checking
    # -------------------

    # The user has tried to provide their own structure of the
    # data, sanity check this as much as possible to try and see
    # if it makes sense.

    @staticmethod
    def check_datashape(source, given_dshape):
        """
        Does the user specified dshape make sense for the given
        source.
        """
        # TODO
        return True

    @staticmethod
    def check_layout(layout):
        """
        Does the user specified layout make sense for the given
        source.
        """
        return isinstance(layout, ChunkedL)

    def repr_data(self):
        return carray.array2string(self.ca)

    def read(self, elt, key):
        """ CArray extension supports reading directly from high-level
        coordinates """
        # disregards elt since this logic is implemented lower
        # level in the Cython extension _getrange
        return self.ca.__getitem__(key)

    def __repr__(self):
        return 'CArray(ptr=%r)' % id(self.ca)

#------------------------------------------------------------------------
# Chunked Columns
#------------------------------------------------------------------------

class CTableSource(ByteProvider):
    """ Chunked array is the default storage engine for Blaze arrays
    when no layout is specified. """

    read_capabilities  = CHUNKED
    write_capabilities = CHUNKED
    access_capabilities = ACCESS_ALLOC

    def __init__(self, data, rootdir=None):
        """ CArray object passed directly into the constructor,
        ostensibly this is just a thin wrapper that consumes a
        reference.
        """
        self.ca = carray.ctable(np.empty(data, dtype="i4,f8"))

    # Descriptors
    # -----------

    def read_desc(self):
        return CArrayDataDescriptor('carray_dd', self.ca.nbytes, self.ca)

    @staticmethod
    def infer_datashape(source):
        """
        The user has only provided us with a Python object ( could be
        a buffer interface, a string, a list, list of lists, etc) try
        our best to infer what the datashape should be in the context of
        what it would mean as a CTable.
        """
        if isinstance(source, np.ndarray):
            return from_numpy(source.shape, source.dtype)
        elif isinstance(source, list):
            # TODO: um yeah, we'd don't actually want to do this
            cast = np.array(source)
            return from_numpy(cast.shape, cast.dtype)
        else:
            return dynamic

    def repr_data(self):
        return carray.table2string(self.ca)

    def read(self, elt, key):
        """ CArray extension supports reading directly from high-level
        coordinates """
        # disregards elt since this logic is implemented lower
        # level in the Cython extension _getrange
        return self.ca.__getitem__(key)

    def __repr__(self):
        return 'CTable(ptr=%r)' % id(self.ca)
