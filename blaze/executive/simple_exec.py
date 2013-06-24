from __future__ import absolute_import

'''This module contains a sample, rather naive, executor of blaze
functions over blaze arrays.

The blaze function is described by a BlazeFuncDescriptor. The result
will be placed in a concrete target data_descriptor.
'''

from itertools import product as it_product
import ctypes
from ..py3help import izip

from ..datashape.util import to_ctypes

class _Executor(object):
    """
    A simple executor class that is able to convert a BlazeFunc
    DataDescriptor into a NumPy DataDescriptor
    """
    def __init__(self, dd, iter_dims=1):
        res_ds = dd.dshape
        total_dims = len(res_ds) - 1
        lift_dims = total_dims - iter_dims

        # one reader per arg
        readers = [arr.arr._data.element_reader(iter_dims)
                   for arr in dd.args]

        tree = dd.kerneltree.fuse()
        newkernel = tree.kernel.lift(max(1, lift_dims), 'C')
        cfunc = newkernel.ctypes_func
        types = [to_ctypes(ds.measure) for ds in newkernel.dshapes]

        if lift_dims < 1 :
            arg_structs = [arg_type._type_(None, (1,))
                           for arg_type in cfunc.argtypes]
        else:
            shapes = [arr.arr.dshape.shape[iter_dims:]
                      for arr in dd.args]
            shapes.append(res_ds.shape[iter_dims:])

            arg_structs = [arg_type._type_(None, shape)
                           for arg_type, shape in izip(cfunc.argtypes,
                                                       shapes)]

        self.cfunc = cfunc # kernel to call...
        self.readers = readers # readers for inputs
        self.arg_structs = arg_structs # ctypes args
        self.res_dshape = res_ds # shape of the results
        self.outer_dims = res_ds[:iter_dims] # shape with elements
        self.inner_dims = res_ds[iter_dims:]
        self.c_types = types

    def _patch(self, struct, value, typ=None):
        if typ is None:
            typ = struct.e0._type_
        struct.e0 = ctypes.cast(value, ctypes.POINTER(typ))

    def run_append(self, dd):
        f = self.cfunc
        arg_s = self.arg_structs
        r = self.readers
        with dd.element_appender() as dst:
            for element in it_product(*[xrange(x) for x in self.outer_dims]):
                for struct, reader, typ in izip(arg_s[:-1], r,
                                                self.c_types[:-1]):
                    self._patch(struct, reader.read_single(element), typ)

                with dst.buffered_ptr() as dst_buff:
                    self._patch(arg_s[-1], dst_buff, self.c_types[-1])
                    f(*arg_s)

    def run_write(self, dd):
        f = self.cfunc
        arg_s = self.arg_structs
        r = self.readers
        dst = dd.element_writer(len(self.outer_dims))
        for element in it_product(*[xrange(x) for x in self.outer_dims]):
            for struct, reader, typ in izip(arg_s[:-1], r, self.c_types[:-1]):
                self._patch(struct, reader.read_single(element), typ)
            with dst.buffered_ptr(element) as dst_buf:
                self._patch(arg_s[-1], dst_buf, self.c_types[-1])
                f(*arg_s)

class _CompleteExecutor(object):
    """
    A simple executor class that is able to convert a BlazeFunc
    DataDescriptor into a MemBuf DataDescriptor
    """
    def __init__(self, dd):
        res_ds = dd.dshape
        total_dims = len(res_ds) - 1
        lift_dims = total_dims

        tree = dd.kerneltree.fuse()
        newkernel = tree.kernel.lift(lift_dims, 'C')
        cfunc = newkernel.ctypes_func
        types = [to_ctypes(ds.measure) for ds in newkernel.dshapes]
        # one reader per arg
        readers = [arr.arr._data.element_reader(0)
                   for arr in dd.args]
        shapes = [arr.arr.dshape.shape
                  for arr in dd.args]
        shapes.append(res_ds.shape)
        # Will patch the pointer to data later..
        arg_structs = [arg_type._type_(None, shape)
                       for arg_type, shape in izip(cfunc.argtypes, shapes)]

        self.cfunc = cfunc # kernel to call...
        self.readers = readers # readers for inputs
        self.arg_structs = arg_structs # ctypes args
        self.res_dshape = res_ds # shape of the results
        self.inner_dims = res_ds
        self.c_types = types

    def _patch(self, struct, value, typ=None):
        if typ is None:
            typ = struct.e0._type_
        struct.e0 = ctypes.cast(value, ctypes.POINTER(typ))

    def run_write(self, dd):
        f = self.cfunc
        arg_s = self.arg_structs
        r = self.readers
        dst = dd.element_writer(0)
        element = ()
        for struct, reader, typ in izip(arg_s[:-1], r, self.c_types[:-1]):
            self._patch(struct, reader.read_single(element), typ)
        with dst.buffered_ptr(element) as dst_buf:
            self._patch(arg_s[-1], dst_buf, self.c_types[-1])
            f(*arg_s)


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

def simple_execute_write(in_dd, out_dd, iter_dims=None):
    if iter_dims is None:
        ex = _CompleteExecutor(in_dd)
        ex.run_write(out_dd)
    else:
        ex = _Executor(in_dd, iter_dims)
        ex.run_write(out_dd)

def simple_execute_append(in_dd, out_dd, iter_dims=1):
    assert(iter_dims==1) # append can only work this way ATM
    ex = _Executor(in_dd)
    ex.run_append(out_dd)
