from __future__ import absolute_import
# This file defines the Concrete Array --- a leaf node in the expression graph
#
# A concrete array is constructed from a Data Descriptor Object which handles the
#  indexing and basic interpretation of bytes
#

import sys
import ctypes

from . import blz
import numpy as np
from .datashape import dshape, coretypes as T
from .datashape.util import to_ctypes
from .datadescriptor import (IDataDescriptor,
                             data_descriptor_from_ctypes,
                             DyNDDataDescriptor)
from .executive import simple_execute_write
from ._printing import array2string as _printer
from .py2help import exec_
from .bkernel import bmath

# An Array contains:
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
        return self._data.dshape

    @property
    def deferred(self):
        return self._data.deferred

    @property
    def persistent(self):
        return self._data.persistent

    def __iter__(self):
        return self._data.__iter__()

    def __getitem__(self, key):
        return Array(self._data.__getitem__(key))

    def __setitem__(self, key, val):
        self._data.__setitem__(key, val)

    def __len__(self):
        return self._data.dshape[0]

    def __str__(self):
        if hasattr(self._data, '_printer'):
            return self._data._printer()
        return _printer(self._data)

    def __repr__(self):
        pre = 'array('
        post =  ',\n' + ' '*len(pre) + "dshape='" + str(self.dshape) + "'" + ')'
        if hasattr(self._data, '_printer'):
            body = self._data._printer()
        else:
            body = _printer(self._data,
                              separator=', ',
                              prefix=' '*len(pre))

        return pre + body + post

    def __init__(self, data, axes=None, labels=None, user={}):
        if not isinstance(data, IDataDescriptor):
            raise TypeError(('Constructing a blaze array directly '
                            'requires a data descriptor, not type '
                            '%r') % (type(data)))
        self._data = data
        self.axes = axes or [''] * (len(self._data.dshape) - 1)
        self.labels = labels or [None] * (len(self._data.dshape) - 1)
        self.user = user

        # In the case of dynd arrays, inject the record attributes.
        # This is a hack to help get the blaze-web server onto blaze arrays.
        if isinstance(data, DyNDDataDescriptor):
            ms = data.dshape[-1]
            if isinstance(ms, T.Record):
                props = {}
                for name in ms.names:
                    props[name] = _named_property(name)
                self.__class__ = type('blaze.Array', (Array,), props)

        # Need to inject attributes on the Array depending on dshape
        # attributes, in cases other than Record

def _named_property(name):
    @property
    def getprop(self):
        return Array(DyNDDataDescriptor(getattr(self._data.dynd_arr(), name)))
    return getprop

_template = """
def __{name}__(self, other):
    return bmath.{name}(self, other)
"""

def inject_special(names):
    dct = {'bmath':bmath}
    for name in names:
        newname = '__%s__' % name
        exec_(_template.format(name=name), dct)
        setattr(Array, newname, dct[newname])

inject_special(['add', 'sub', 'mul', 'truediv', 'mod', 'floordiv',
                'eq', 'ne', 'gt', 'ge', 'le', 'lt', 'div'])


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

