from __future__ import absolute_import
# This file defines the Concrete Array --- a leaf node in the expression graph
#
# A concrete array is constructed from a Data Descriptor Object which handles the
#  indexing and basic interpretation of bytes
#

from .datashape import dshape
from .datadescriptor import DataDescriptor

# An Array is a
#   DataDescriptor
#       Sequence of Bytes (where are the bytes)
#       Index Object (how do I get to them)
#       Data Shape Object (what are the bytes? how do I interpret them)
#
#   axis and dimension labels 
#   user-defined meta-data (whatever are needed --- provenance propagation)
class Array(object):

    @property
    def dshape(self):
        return self._dshape

    @property
    def shape(self):
        return self._dshape.shape

    @property
    def nd(self):
        return len(self._dshape, shape)

    def __iter__(self, iter):
        return self.data.__iter__()

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __init__(self, data, axes=None, labels=None, user={}):
        assert isinstance(data, DataDescriptor)
        self.data = data
        self.axes = axes or [''] * self.data.nd
        self.labels = labels or [None] * self.data.nd
        self.user = user

        # Need to inject attributes on the Array depending on dshape attributes

"""
These should be functions

    @staticmethod
    def fromfiles(list_of_files, converters):
        raise NotImplementedError

    @staticmethod
    def fromfile(file, converter):
        raise NotImplementedError

    @staticmethod
    def frombuffers(list_of_buffers, converters):
        raise NotImplementedError

    @staticmethod
    def frombuffer(buffer, converter):
        raise NotImplementedError

    @staticmethod
    def fromobjects():
        raise NotImplementedError

    @staticmethod
    def fromiterator(buffer):
        raise NotImplementedError

"""
        
