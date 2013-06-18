from __future__ import absolute_import

'''This module contains a sample, rather naive, executor of blaze
functions over blaze arrays.

The blaze function is described by a BlazeFuncDescriptor. The result
will be placed in a concrete target data_descriptor.
'''

from itertools import izip, product as it_product
import ctypes

class _Executor(object):
    """
    A simple executor class that is able to convert a BlazeFunc
    DataDescriptor into a NumPy DataDescriptor
    """
    def __init__(self, dd, iter_dims=1):
        res_ds = dd.dshape
        total_dims = len(res_ds) - 1
        lift_dims = total_dims - iter_dims

        cfunc = dd.kerneltree._fused.kernel.lift(lift_dims, 'C').ctypes_func
        # one reader per arg
        readers = [arr.arr._data.element_reader(iter_dims)
                   for arr in dd.args]
        shapes = [arr.arr.dshape.shape[iter_dims:]
                  for arr in dd.args]
        shapes.append(res_ds.shape[iter_dims:])
        arg_structs = [arg_type._type_(None, shape)
                       for arg_type, shape in izip(cfunc.argtypes, shapes)]

        self.cfunc = cfunc # kernel to call...
        self.readers = readers # readers for inputs
        self.arg_structs = arg_structs # ctypes args
        self.res_dshape = res_ds # shape of the results
        self.outer_dims = res_ds[:iter_dims] # shape with elements
        self.inner_dims = res_ds[iter_dims:]

    def _patch(self, struct, value):
        struct.e0 = ctypes.cast(value, ctypes.POINTER(struct.e0._type_))

    def run_append(self, dd):
        f = self.cfunc
        arg_s = self.arg_structs
        r = self.readers
        with dd.element_appender() as dst:
            for element in it_product(*[xrange(x) for x in self.outer_dims]):
                for struct, reader in izip(arg_s[:-1], r):
                    self._patch(struct, reader.read_single(element))

                with dst.buffered_ptr() as dst_buff:
                    self._patch(arg_s[-1], dst_buff)
                    f(*arg_s)

    def run_write(self, dd):
        f = self.cfunc
        arg_s = self.arg_structs
        r = self.readers
        dst = dd.element_writer(len(self.outer_dims))
        for element in it_product(*[xrange(x) for x in self.outer_dims]):
            for struct, reader in izip(arg_s[:-1], r):
                self._patch(struct, reader.read_single(element))
            with dst.buffered_ptr(element) as dst_buf:
                self._patch(arg_s[-1], dst_buf)
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
        iter_dims = _iter_dims_heuristic(in_dd, out_dd)
    ex = _Executor(in_dd)
    ex.run_write(out_dd)

def simple_execute_append(in_dd, out_dd, iter_dims=1):
    assert(iter_dims==1) # append can only work this way ATM
    ex = _Executor(in_dd)
    ex.run_append(out_dd)
