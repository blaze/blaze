from __future__ import absolute_import, division, print_function

from dynd import nd
from glob import glob
from itertools import chain
from datashape import dshape, var
from datashape.predicates import isdimension

from .core import DataDescriptor
from .. import compatibility
from ..utils import get, ndget

__all__ = 'Concat', 'Stack'


class Concat(DataDescriptor):
    def __init__(self, descriptors):
        assert all(isdimension(ddesc.dshape[0]) for ddesc in descriptors)
        self.descriptors = descriptors

    @property
    def dshape(self):
        return var * self.descriptors[0].dshape.subarray(1)

    def _iter(self):
        return chain.from_iterable(self.descriptors)

    def _chunks(self, **kwargs):
        return (chunk for dd in self.descriptors
                      for chunk in dd.chunks(**kwargs))

    def _get_py(self, key):
        if not isinstance(key, tuple):
            return get(key, iter(self))

        result = get(key[0], iter(self))
        if isinstance(key[0], (list, slice)):
            return (ndget(key[1:], row) for row in result)
        else:
            return ndget(key[1:], result)


class Stack(DataDescriptor):
    def __init__(self, descriptors):
        self.descriptors = descriptors

    @property
    def dshape(self):
        return len(self.descriptors) * self.descriptors[0].dshape

    def _iter(self):
        return (dd.as_py() for dd in self.descriptors)

    def chunks(self, **kwargs):
        return (dd.as_dynd() for dd in self.descriptors)

    def _get_py(self, key):
        if isinstance(key, tuple):
            result = get(key[0], self.descriptors)
            if isinstance(key[0], (list, slice)):
                return (s._get_py(key[1:]) for s in result)
            else:
                return result._get_py(key[1:])
        else:
            result = get(key, self.descriptors)
            if isinstance(key, (list, slice)):
                return (s.as_py() for s in result)
            else:
                return result.as_py()
