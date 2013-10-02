from __future__ import absolute_import

'''This module contains a sample, rather naive, executor of blaze
functions over blaze arrays.

The blaze function is described by a BlazeFuncDeprecatedDescriptor. The result
will be placed in a concrete target data_descriptor.
'''

from itertools import product as it_product
import ctypes
import operator
import logging

from ..py2help import izip, reduce, xrange
from ..datashape.util import to_ctypes


def _chunk_size_gen(total, chunk):
    a = 0
    while a + chunk < total:
        yield (a, chunk)
        a += chunk

    yield (a, total - a)


def _chunk_split(dims, chunk_size):
    # returns for each chunk its "index" as well as its actual chunk size.
    if len(dims) > 0:
        helper_it = it_product(*([xrange(x) for x in dims[:-1]] +
                                 [_chunk_size_gen(dims[-1], chunk_size)]))

        for i in helper_it:
            yield i[:-1]+(i[-1][0],), i[-1][1]
    else:
        yield tuple(), 1


class _Executor(object):
    """
    A simple executor class that is able to convert a BlazeFuncDeprecated
    DataDescriptor into a raw memory DataDescriptor
    """

    def __init__(self, dd, iter_dims=1):
        res_ds = dd.dshape
        # the amount of dimensions to lift the kernel.
        # this will be the dimensions of the datashape minus the
        # dimensions we want to iterate on, plus one because the
        # inner dimension of iteration will be chunked.
        # note that len(res_ds) is len(res_ds.shape) + 1 (due to
        # the measure).
        lift_dims = len(res_ds) - iter_dims

        # generate the specialization.
        tree = dd.kerneltree.fuse()
        newkernel = tree.kernel.lift(lift_dims, 'C')
        cfunc = newkernel.ctypes_func


        # one reader per arg
        readers = [arr.arr._data.element_reader(iter_dims)
                   for arr in dd.args]

        ptr_types = [ctypes.POINTER(to_ctypes(ds.measure)) for ds in newkernel.dshapes]

        shapes = [arr.arr.dshape.shape[iter_dims:]
                  for arr in dd.args]
        shapes.append(res_ds.shape[iter_dims:])

        # fill the attributes
        # cfunc -> the kernel
        # readers -> element_readers for the inputs
        # arg_shapes -> the shapes of the inputs and the outputs as expected
        #               by the kernel, but missing the outer dimension (that
        #               will be the chunk size
        # outer_dims -> shape to be used by the iteration.
        self.cfunc = cfunc
        self.readers = readers
        self.ptr_types = ptr_types
        self.arg_shapes = shapes
        self.outer_dims = tuple(operator.index(i) for i in res_ds[:iter_dims])


    def run_append(self, dd, chunk_size=1):
        kernel = self.cfunc
        readers = self.readers
        ptr_types = self.ptr_types
        arg_shapes = self.arg_shapes
        kernel_types = [at._type_ for at in kernel.argtypes]
        byref = ctypes.byref
        cast = ctypes.cast

        with dd.element_appender() as dst:
            for element, chunk_size in _chunk_split(self.outer_dims, chunk_size):
                ptrs = [r.read_single(element, count=chunk_size) for r in readers]
                with dst.buffered_ptr(count=chunk_size) as dst_buff:
                    ptrs += [dst_buff]
                    args = [t(cast(p, pt), (chunk_size, ) + s) 
                            for t,pt,p,s in izip (kernel_types, ptr_types, ptrs,
                                                  arg_shapes)]

                    kernel(*[byref(x) for x in args])


    def run_write(self, dd, chunk_size=1):
        kernel = self.cfunc
        readers = self.readers
        ptr_types = self.ptr_types
        arg_shapes = self.arg_shapes
        kernel_types = [at._type_ for at in kernel.argtypes]
        byref = ctypes.byref
        cast = ctypes.cast

        dst = dd.element_writer(len(self.outer_dims))
        for element, chunk_size in _chunk_split(self.outer_dims, chunk_size):
            ptrs = [r.read_single(element, count=chunk_size) for r in readers]
            with dst.buffered_ptr(element, count=chunk_size) as dst_buff:
                ptrs += [dst_buff]
                args = [t(cast(p, pt), (chunk_size,) + s)
                        for t, pt,  p, s in izip(kernel_types, ptr_types, ptrs, 
                                                 arg_shapes)]
                kernel(*[byref(x) for x in args])


def _iter_dims_heuristic(in_dd, out_dd):
    '''this function could use information on in_dd and out_dd in order to
    select the degree of lifting on the kernel. Right now this is by a
    means of selecting the number of iterations to handle outside of
    the llvm functions. That would mean bigger buffers as well and
    less opportunities to multithread.

    ATM it is not very interesting as it is limited by number of
    dimensions. In the future it may also involve tiling (or at least
    grouping by spans over the outer dimension.

    Right now execute everything in one go by default

    '''

    return 0


def simple_execute_write(in_dd, out_dd, iter_dims=0, chunk=1):
    ex = _Executor(in_dd, iter_dims)
    ex.run_write(out_dd, chunk)


def simple_execute_append(in_dd, out_dd, iter_dims=1, chunk=1):
    assert(iter_dims==1) # append can only work this way ATM
    ex = _Executor(in_dd)
    ex.run_append(out_dd, chunk)
