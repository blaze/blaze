from __future__ import absolute_import

from . import IDataDescriptor, Capabilities

def blaze_func_iter(bfd, noiter_dims):
    args = bfd.args
    dim_size = 1
    iters = []
    for a, noiter in zip(args, noiter_dims):
        if noiter:
            iters.append(a)
        else:
            # TODO handle streaming dimension with no __len__
            arg_dim_size = len(a)
            if dim_size == 1:
                dim_size = arg_dim_size
            elif dim_size != arg_dim_size:
                raise BroadcastError(('Cannot broadcast dimensions of ' +
                                'size %d and %d together') % (dim_size, arg_dim_size))
            iters.append(a.__iter__())
    # TODO continue...


class BlazeFuncDeprecatedDescriptor(IDataDescriptor):
    """
    Data descriptor for blaze.bkernel.BlazeFunc

    Attributes:
    ===========
    kerneltree: blaze.bkernel.kernel_tree.KernelTree
        deferred expression DAG/tree

    outdshape: DataShape
        result type

    argmap: { blaze.bkernel.kernel_tree.Argument : Array }
        Keeps track of concrete input arrays
    """

    _args = None
    deferred = True

    def __init__(self, kerneltree, outdshape, argmap):
        self.kerneltree = kerneltree
        self.outdshape = outdshape
        self.argmap = argmap

    def _reset_args(self):
        from blaze.compute.bkernel.kernel_tree import find_unique_args
        unique_args = []
        find_unique_args(self.kerneltree, unique_args)
        self._args = [self.argmap[argument] for argument in unique_args]

    @property
    def capabilities(self):
        """The capabilities for the blaze function data descriptor."""
        return Capabilities(
            immutable = True,
            deferred = True,
            # persistency is not supported yet
            persistent = False,
            appendable = False,
            )

    @property
    def args(self):
        if self._args is None:
            self._reset_args()
        return self._args

    @property
    def isfused(self):
        from blaze.compute.bkernel.kernel_tree import Argument
        return all(isinstance(child, Argument) for child in self.kerneltree.children)

    def fuse(self):
        if not self.isfused:
            return self.__class__(self.kerneltree.fuse(), self.outdshape)
        else:
            return self

    def _printer(self):
        return str(self.kerneltree)

    @property
    def dshape(self):
        return self.outdshape

    def __iter__(self):
        # Figure out how the outermost dimension broadcasts, by
        # subtracting the rank sizes of the blaze func elements from
        # the argument dshape sizes
        broadcast_dims = [len(a.dshape) - len(e.dshape)
                        for a, e in zip(self.args, self.kernel_elements)]
        ndim = max(broadcast_dims)
        if ndim > 1:
            # Do a data descriptor-level broadcasting
            noiter_dims = [x == 0 for x in broadcast_dim]
            return blaze_func_iter(self, noiter_dims)
        elif ndim == 1:
            # Do an element-level broadcasting
            raise NotImplemented
        else:
            raise IndexError('Cannot iterate over a scalar')

    def __getitem__(self, key):
        raise NotImplementedError

