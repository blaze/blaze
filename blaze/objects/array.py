"""This file defines the Concrete Array --- a leaf node in the expression graph

A concrete array is constructed from a Data Descriptor Object which handles the
 indexing and basic interpretation of bytes
"""

from __future__ import absolute_import, division, print_function

import datashape

from ..compute.ops import ufuncs
from .. import compute

from ..datadescriptor import (I_DDesc, DeferredDescriptor)
from ..io import _printing


class Array(object):
    """An Array contains:

        DDesc
        Sequence of Bytes (where are the bytes)
        Index Object (how do I get to them)
        Data Shape Object (what are the bytes? how do I interpret them)
        axis and dimension labels
        user-defined meta-data (whatever are needed --- provenance propagation)
    """
    def __init__(self, data, axes=None, labels=None, user={}):
        if not isinstance(data, I_DDesc):
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
        injected_props = {}
        # This is a hack to help get the blaze-web server onto blaze arrays.
        ds = data.dshape
        ms = ds[-1] if isinstance(ds, datashape.DataShape) else ds
        if isinstance(ms, datashape.Record):
            for name in ms.names:
                injected_props[name] = _named_property(name)

        # Need to inject attributes on the Array depending on dshape
        # attributes, in cases other than Record
        if data.dshape in [datashape.dshape('int32'), datashape.dshape('int64')]:
            def __int__(self):
                # Evaluate to memory
                e = compute.eval.eval(self)
                return int(e._data.dynd_arr())
            injected_props['__int__'] = __int__
        elif data.dshape in [datashape.dshape('float32'), datashape.dshape('float64')]:
            def __float__(self):
                # Evaluate to memory
                e = compute.eval.eval(self)
                return float(e._data.dynd_arr())
            injected_props['__float__'] = __float__
        elif ms in [datashape.complex_float32, datashape.complex_float64]:
            if len(data.dshape) == 1:
                def __complex__(self):
                    # Evaluate to memory
                    e = compute.eval.eval(self)
                    return complex(e._data.dynd_arr())
                injected_props['__complex__'] = __complex__
            injected_props['real'] = _ufunc_to_property(ufuncs.real)
            injected_props['imag'] = _ufunc_to_property(ufuncs.imag)

        if injected_props:
            self.__class__ = type('Array', (Array,), injected_props)


    @property
    def dshape(self):
        return self._data.dshape

    @property
    def deferred(self):
        return self._data.capabilities.deferred


    def __array__(self):
        import numpy as np

        # TODO: Expose PEP-3118 buffer interface

        if hasattr(self._data, "__array__"):
            return np.array(self._data)

        return np.array(self._data.dynd_arr())

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
        raise IndexError('Scalar blaze arrays have no length')

    def __nonzero__(self):
        # For Python 2
        if len(self.dshape.shape) == 0:
            # Evaluate to memory
            e = compute.eval.eval(self)
            return bool(e._data.dynd_arr())
        else:
            raise ValueError("The truth value of an array with more than one "
                             "element is ambiguous. Use a.any() or a.all()")

    def __bool__(self):
        # For Python 3
        if len(self.dshape.shape) == 0:
            # Evaluate to memory
            e = compute.eval.eval(self)
            return bool(e._data.dynd_arr())
        else:
            raise ValueError("The truth value of an array with more than one "
                             "element is ambiguous. Use a.any() or a.all()")

    def __str__(self):
        if hasattr(self._data, '_printer'):
            return self._data._printer()
        return _printing.array_str(self)

    def __repr__(self):
        if hasattr(self._data, "_printer_repr"):
            return self._data._printer_repr()
        return _printing.array_repr(self)


def _named_property(name):
    @property
    def getprop(self):
        return Array(self._data.getattr(name))
    return getprop


def _ufunc_to_property(uf):
    @property
    def getprop(self):
        return uf(self)
    return getprop


def binding(f):
    def binder(self, *args):
        return f(self, *args)
    return binder


def __rufunc__(f):
    def __rop__(self, other):
        return f(other, self)
    return __rop__


def _inject_special_binary(names):
    for ufunc_name, special_name in names:
        ufunc = getattr(ufuncs, ufunc_name)
        setattr(Array, '__%s__' % special_name, binding(ufunc))
        setattr(Array, '__r%s__' % special_name, binding(__rufunc__(ufunc)))


def _inject_special(names):
    for ufunc_name, special_name in names:
        ufunc = getattr(ufuncs, ufunc_name)
        setattr(Array, '__%s__' % special_name, binding(ufunc))


_inject_special_binary([
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
    ('divide', 'div'),
    ('bitwise_and', 'and'),
    ('bitwise_or', 'or'),
    ('bitwise_xor', 'xor'),
    ('power', 'pow'),
    ])
_inject_special([
    ('bitwise_not', 'invert'),
    ('negative', 'neg'),
    ])


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

