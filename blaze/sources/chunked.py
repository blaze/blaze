"""
Wrappers around chunked arrays ( carray ) supporting both
in-memory and on-disk storage.
"""

from blaze import carray

from blaze.byteprovider import ByteProvider
from blaze.byteproto import CONTIGUOUS, CHUNKED, STREAM, ACCESS_ALLOC
from blaze.datadescriptor import CArrayDataDescriptor
from blaze.datashape.coretypes import dynamic, from_numpy, to_numpy

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

    def __init__(self, ca, rootdir=None):
        """ CArray object passed directly into the constructor,
        ostensibly this is just a thin wrapper that consumes a
        reference.
        """
        self.ca = carray.carray(ca, rootdir=rootdir)

    def read_desc(self):
        return CArrayDataDescriptor('carray_dd', self.ca.nbytes, self.ca)

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
        elif isinstance(source, list):
            # TODO: um yeah, we'd don't actually want to do this
            cast = np.array(source)
            return from_numpy(cast.shape, cast.dtype)
        else:
            return dynamic

    @staticmethod
    def check_datashape(source, given_dshape):
        """
        Does the user specified dshape make sense for the given
        source.
        """
        # TODO
        return True

    def repr_data(self):
        return carray.array2string(self.ca)

    @classmethod
    def empty(self, dshape):
        """ Create a CArraySource from a datashape specification,
        downcasts into Numpy dtype and shape tuples if possible
        otherwise raises an exception.
        """
        shape, dtype = from_numpy(dshape)
        return CArraySource(carray([], dtype))

    def read(self, elt, key):
        """ CArray extension supports reading directly from high-level
        coordinates """
        # disregards elt since this logic is implemented lower
        # level in the Cython extension _getrange
        return self.ca.__getitem__(key)

    def __repr__(self):
        return 'CArray(ptr=%r)' % id(self.ca)
