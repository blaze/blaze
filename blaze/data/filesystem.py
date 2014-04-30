from __future__ import absolute_import, division, print_function

from dynd import nd
from glob import glob
from itertools import chain
from datashape import dshape, Var

from .core import DataDescriptor
from .. import compatibility

__all__ = 'Files',

class Files(DataDescriptor):
    immutable = True
    deferred = False
    appendable = False
    remote = False
    persistent = True

    def __init__(self, files, descriptor, subdshape=None, schema=None,
            open=open):
        if isinstance(files, compatibility._strtypes):
            files = glob(files)
        self.filenames = files

        self.open = open

        self.descriptor = descriptor
        if schema and not subdshape:
            subdshape = Var() * schema
        self.subdshape = dshape(subdshape)

    @property
    def dshape(self):
        if isinstance(self.subdshape[0], Var):
            return self.subdshape
        else:
            return Var() * self.subdshape

    def _iter(self):
        return chain.from_iterable(self.descriptor(fn,
                                                   dshape=self.subdshape,
                                                   open=self.open)
                                    for fn in self.filenames)
