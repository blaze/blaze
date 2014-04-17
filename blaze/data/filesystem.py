from __future__ import absolute_import, division, print_function

from dynd import nd
from glob import glob
from itertools import chain

from .core import DataDescriptor
from .. import py2help

__all__ = 'Files',

class Files(DataDescriptor):
    immutable = True
    deferred = False
    appendable = False
    remote = False
    persistent = True

    def __init__(self, files, descriptor, schema=None):
        if isinstance(files, py2help._strtypes):
            files = glob(files)
        self.filenames = files

        self.descriptor = descriptor
        self._schema = schema

    def _iter(self):
        return chain.from_iterable(self.descriptor(fn, schema=self.schema)
            for fn in self.filenames)
        return iter(self.storage)
