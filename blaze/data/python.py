from __future__ import absolute_import, division, print_function

from dynd import nd

from .core import DataDescriptor
from ..utils import ndget

class Python(DataDescriptor):
    def __init__(self, storage=None, schema=None, dshape=None, **kwargs):
        self.storage = storage if storage is not None else []
        self._schema = schema
        self._dshape = dshape

    def _extend(self, seq):
        self.storage.extend(seq)

    def _iter(self):
        return iter(self.storage)

    def _get_py(self, key):
        return ndget(key, self.storage)

    def as_py(self):
        return tuple(self.storage)
