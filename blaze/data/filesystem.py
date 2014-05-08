from __future__ import absolute_import, division, print_function

from dynd import nd
from glob import glob
from itertools import chain
from datashape import dshape, var
from datashape.predicates import isdimension

from .core import DataDescriptor
from .. import compatibility

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


class Stack(DataDescriptor):
    def __init__(self, descriptors):
        self.descriptors = descriptors

    @property
    def dshape(self):
        return len(self.descriptors) * self.descriptors[0].dshape

    def _iter(self):
        return (dd.as_py() for dd in self.descriptors)

    def _chunks(self, **kwargs):
        return (dd.as_dynd() for dd in self.descriptors)

