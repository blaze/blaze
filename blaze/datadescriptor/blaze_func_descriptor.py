from __future__ import absolute_import

from . import DataDescriptor, IGetElement, IElementIter

class BlazeFuncDescriptor(DataDescriptor):

    def __init__(self, kerneltree, outdshape, args):
        self.kerneltree = kerneltree
        self.outdshape = outdshape
        self.args = args

        # This is a dictionary of unique names on the nodes of the 
        #  kernel tree and entry numbers in "args"
        #  A general kernel tree might suggest many arguments however
        #  in practice most of these arguments come from particular sources 
        # The same array might show up multiple times in the arglist
        #  New entries will over-write odl entries. 
        argmap = {}
        for i, arg in enumerate(args):
            argmap[arg.data.unique_name]=i
        self.argmap = argmap

    @property
    def dshape(self):
        return self.outdshape

