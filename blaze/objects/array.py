"""This file defines the Concrete Array --- a leaf node in the expression graph

A concrete array is constructed from a Data Descriptor Object which handles the
 indexing and basic interpretation of bytes
"""

from __future__ import absolute_import, division, print_function

from datashape import coretypes as T
from blaze.compute.expr import dump
from blaze.compute.ops import ufuncs

from ..datadescriptor import (IDataDescriptor,
                              DyNDDataDescriptor,
                              DeferredDescriptor)
from ..io import _printing


class Array(object):
    """An Array contains:

        DataDescriptor
        Sequence of Bytes (where are the bytes)
        Index Object (how do I get to them)
        Data Shape Object (what are the bytes? how do I interpret them)
        axis and dimension labels
        user-defined meta-data (whatever are needed --- provenance propagation)
    """
    def __init__(self, data, axes=None, labels=None, user={}):
        if not isinstance(data, IDataDescriptor):
            raise TypeError(('Constructing a blaze array directly '
                            'requires a data descriptor, not type '
                            '%r') % (type(data)))
        self._data = data
        self.axes = axes or [''] * (len(self._data.dshape) - 1)
        self.labels = labels or [None] * (len(self._data.dshape) - 1)
        self.user = user
        self.expr = None

        if isinstance(data, DeferredDescriptor):
            # NOTE: we need 'expr' on the Array to perform dynamic programming:
            #       Two concrete arrays should have a single Op! We cannot
            #       store this in the data descriptor, since there are many
            self.expr = data.expr # hurgh

        # Inject the record attributes.
        # This is a hack to help get the blaze-web server onto blaze arrays.
        ms = data.dshape
        if isinstance(ms, T.DataShape): ms = ms[-1]
        if isinstance(ms, T.Record):
            props = {}
            for name in ms.names:
                props[name] = _named_property(name)
            self.__class__ = type('blaze.Array', (Array,), props)

        # Need to inject attributes on the Array depending on dshape
        # attributes, in cases other than Record

    @property
    def dshape(self):
        return self._data.dshape

    @property
    def deferred(self):
        return self._data.capabilities.deferred

    def view(self):
        if not self.capabilities.deferred:
            raise ValueError("Cannot call 'view' on a concrete array")

        term, context = self.expr
        ipython = False
        try:
            ipython = __IPYTHON__
        except NameError:
            pass

        return dump(term, ipython=ipython)

    def __array__(self):
        import numpy as np

        # TODO: Expose PEP-3118 buffer interface

        if hasattr(self._data, "__array__"):
            return np.array(self._data)

        raise NotImplementedError(self._data)

    def __iter__(self):
        return self._data.__iter__()

    def __getitem__(self, key):
        return Array(self._data.__getitem__(key))

    def __setitem__(self, key, val):
        self._data.__setitem__(key, val)

    def __len__(self):
        shape = self.dshape.shape
        if shape:
            return shape[0]
        return 1 # 0d

    def __nonzero__(self):
        shape = self.dshape.shape
        if len(self) == 1 and len(shape) <= 1:
            if len(shape) == 1:
                item = self[0]
            else:
                item = self[()]
            return bool(item)
        else:
            raise ValueError("The truth value of an array with more than one "
                             "element is ambiguous. Use a.any() or a.all()")

    def __str__(self):
        return _printing.array_str(self)

    def __repr__(self):
        return _printing.array_repr(self)


def _named_property(name):
    @property
    def getprop(self):
        return Array(self._data.getattr(name))
    return getprop


def binding(f):
    def binder(self, *args):
        return f(self, *args)
    return binder


def __rufunc__(f):
    def __rop__(self, other):
        return f(other, self)
    return __rop__


def _inject_special(names):
    for ufunc_name, special_name in names:
        ufunc = getattr(ufuncs, ufunc_name)
        setattr(Array, '__%s__' % special_name, binding(ufunc))
        setattr(Array, '__r%s__' % special_name, binding(__rufunc__(ufunc)))


_inject_special([
    ('add', 'add'),
    ('subtract', 'sub'),
    ('multiply', 'mul'),
    ('true_divide', 'truediv'),
    ('mod', 'mod'),
    ('floor_divide', 'floordiv'),
    ('equal', 'eq'),
    ('not_equal', 'ne'),
    ('greater', 'gt'),
    ('greater_equal', 'ge'),
    ('less_equal', 'le'),
    ('less', 'lt'),
    ('divide', 'div')])


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

