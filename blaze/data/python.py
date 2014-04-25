from __future__ import absolute_import, division, print_function

from dynd import nd

from .core import DataDescriptor

class Python(DataDescriptor):
    immutable = False
    deferred = False
    appendable = True
    remote = False
    persistent = False

    def __init__(self, storage=None, schema=None, dshape=None):
        self.storage = storage if storage is not None else []
        self._schema = schema
        self._dshape = dshape

    def _extend(self, seq):
        self.storage.extend(seq)

    def _iter(self):
        return iter(self.storage)

    def _getitem(self, key):
        return self.storage[key]

    def as_py(self):
        return self.storage
