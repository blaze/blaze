from __future__ import absolute_import

from .data_descriptor import (IDataDescriptor, IElementReader, IElementReadIter)


from ..blaze_kernels import find_unique_args, Argument

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

class BlazeFuncDescriptor(IDataDescriptor):
    _args = None
    deferred = True

    def __init__(self, kerneltree, outdshape):
        self.kerneltree = kerneltree
        self.outdshape = outdshape

    def _reset_args(self):
        unique_args = []
        find_unique_args(self.kerneltree, unique_args)
        self._args = [argument for argument in unique_args]

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

    def _printer(self):
        return str(self.kerneltree)

    @property
    def dshape(self):
        return self.outdshape

    @property
    def writable(self):
        return False

    @property
    def immutable(self):
        # TODO: If all the args are immutable, the result
        #       is also immutable
        return False

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
        raise NotImplemented

    def element_read_iter(self):
        raise NotImplemented

    def element_reader(self, nindex):
        raise NotImplemented
