from __future__ import absolute_import

from . import DataDescriptor, IGetElement, IElementIter


class BlazeFuncDescriptor(DataDescriptor):

    def __init__(self, kerneltree, outshape, args):
        self.kerneltree = kerneltree
        self.outshape = outshape
        self.args = args

    @property
    def dshape(self):
        return self.outshape

