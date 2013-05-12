from __future__ import absolute_import

from . import (IDataDescriptor, IElementReader, IElementReadIter)


from ..blaze_kernels import find_unique_args, Argument

class BlazeFuncDescriptor(IDataDescriptor):
    _args = None

    def __init__(self, kerneltree, outdshape):
        self.kerneltree = kerneltree
        self.outdshape = outdshape

    def _reset_args(self):
        unique_args = []
        find_unique_args(self.kerneltree, unique_args)
        self._args = [argument.arg for argument in unique_args]

    @property
    def args(self):
        if self._args is None:
            self._reset_args()
        return self._args

    @property
    def isfused(self):
        return all(isinstance(child, Argument) for child in self.kerneltree.children)

    def fuse(self):
        if not self.isfused:
            return self.__class__(self.kerneltree.fuse(), self.outdshape)
        else:
            return self

    # Create a new DataDescriptor 
    def reify(self):
        new = self.fuse()


    @property
    def dshape(self):
        return self.outdshape

    def __iter__(self, ):
        return NotImplemented

    def __getitem__(self, key):
        return NotImplemented

    def element_read_iter(self):
        return NotImplemented

    def element_reader(self, nindex):
        return NotImplemented
