# -*- coding: utf-8 -*-

"""
SciDB data descriptor.
"""

from __future__ import print_function, division, absolute_import

import blaze
from blaze.datadescriptor import IDataDescriptor
from .datatypes import scidb_dshape

import scidbpy

class SciDBDataDesc(IDataDescriptor):
    """
    SciDB data descriptor.
    """

    deferred = True

    def __init__(self, dshape, query):
        """
        Parameters
        ----------

        query: Query
            Query object signalling the SciDB array to be referenced, or the
            (atomic) expression to construct an array.
        """
        self.query = query
        self._dshape = dshape

    @property
    def dshape(self):
        return self.dshape

    @property
    def is_concrete(self):
        return False # TODO:

    @property
    def writable(self):
        return True

    @property
    def immutable(self):
        return True

    # TODO: below

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def dynd_arr(self):
        raise NotImplementedError