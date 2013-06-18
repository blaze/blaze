from __future__ import print_function

# inspired from test_blfuncs.py

from blaze.blfuncs import BlazeFunc
from blaze.datashape import double, complex128 as c128

import blaze
import blaze.blz as blz

from blaze.datashape import to_numpy
from blaze.datadescriptor import (NumPyDataDescriptor,
                                  BLZDataDescriptor)
from blaze.executive import (simple_execute_write,
                             simple_execute_append)

from itertools import izip, product as it_product
import numpy as np
import ctypes


def _add(a,b):
    return a+b

def _mul(a,b):
    return a*b

add = BlazeFunc('add', [ ('f8(f8,f8)', _add),
                         ('c16(c16,c16)', _add)])
mul = BlazeFunc('mul', {(double,)*3: _mul,
                          (c128,)*3: _mul })


#dummy_add = """
#void d_add(double * result,
#           const double * src1,
#           const double * src2) {
#    *result = *src1 + *src2;
#}
#"""
#
#d_add = BlazeFunc('d_add', [('cpp', dummy_add)])


a = blaze.array([[1,2,3]]*10000,dshape=double)
b = blaze.array([[6,5,4]]*10000,dshape=double)

c = add(a,b)
d = mul(c,c)


# now d contains an array with a data-provider representing the operation
# (a+b)*(a+b)
# how to build a concrete array containing the results?
def banner(title=None):
    if title is None:
        print("-"*72)
    else:
        print("-- %s %s" %(title, '-'*(68 - len(title))))

banner("func_ptr")
print(d._data.kerneltree.func_ptr)
banner("ctypes_func")
print(d._data.kerneltree.ctypes_func)
banner()


########################################################################
#  Try to get the kernel to execute in the context of the kernel tree  #
########################################################################
def _convert(c_type, ptr, shape):
    b = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
    cs = ctypes.c_ssize_t * len(shape)
    s = cs(*shape)
    return c_type(b,s)

def _mk_array_c_ref(c_type, ptr, shape):
    b = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
    cs = ctypes.c_ssize_t * len(shape)
    s = cs(*shape)
    return ctypes.byref(c_type(b,s))


def execute_datadescriptor(dd):
    # make a lifted fused func...
    lifted = dd.kerneltree._fused.kernel.lift(2,'C')
    cf = lifted.ctypes_func
    # the actual ctypes function to call
    args = [(ct._type_,
             arr.arr._data.element_reader(0).read_single(()),
             arr.arr.dshape.shape)
            for ct, arr in izip(cf.argtypes[:-1], dd.args)]

    res_dd = NumPyDataDescriptor(np.empty(*to_numpy(dd.dshape)))
    with res_dd.element_writer(0).buffered_ptr(()) as dst_ptr:
        args.append((cf.argtypes[-1]._type_, dst_ptr, res_dd.shape))
        cf_args = [_convert(*foo) for foo in args]
        cf(*[ctypes.byref(x) for x in cf_args])

    return blaze.Array(res_dd)


def execute_datadescriptor_outerdim(dd):
    # only lift by one
    lifted = dd.kerneltree._fused.kernel.lift(1,'C')
    cf = lifted.ctypes_func
    print(dir(cf))
    # element readers for operands
    args = [(ct._type_,
             arr.arr._data.element_reader(1),
             arr.arr.dshape.shape[1:])
            for ct, arr in izip(cf.argtypes[:-1], dd.args)]

    res_dd = NumPyDataDescriptor(np.empty(*to_numpy(dd.dshape)))
    outer_dimension = res_dd.shape[0]
    dst = res_dd.element_writer(1)

    for i in xrange(outer_dimension):
        args_i = [(t, er.read_single((i,)), sh) for t, er, sh in args]
        with dst.buffered_ptr((i,)) as dst_ptr:
            args_i.append((cf.argtypes[-1]._type_, dst_ptr, res_dd.shape[1:]))
            cf_args = [_convert(*foo) for foo in args_i]
            cf(*[ctypes.byref(x) for x in cf_args])

    return blaze.Array(res_dd)

def execute_datadescriptor_ooc(dd, res_name=None):
    # only lift by one
    res_ds = dd.dshape
    res_shape, res_dt = to_numpy(dd.dshape)

    lifted = dd.kerneltree._fused.kernel.lift(1,'C')
    cf = lifted.ctypes_func

    # element readers for operands
    args = [(ct._type_,
             arr.arr._data.element_reader(1),
             arr.arr.dshape.shape[1:])
            for ct, arr in izip(cf.argtypes[:-1], dd.args)]

    res_dd = BLZDataDescriptor(blz.zeros((0,) + res_shape[1:],
                                         dtype = res_dt,
                                         rootdir = res_name))

    res_ct = ctypes.c_double*3
    res_buffer = res_ct()
    res_buffer_entry = (cf.argtypes[-1]._type_,
                        ctypes.pointer(res_buffer),
                        res_shape[1:])
    with res_dd.element_appender() as ea:
        for i in xrange(res_shape[0]):
            args_i = [(t, er.read_single((i,)), sh)
                      for t, er, sh in args]
            args_i.append(res_buffer_entry)
            cf_args = [_convert(*foo) for foo in args_i]
            cf(*[ctypes.byref(x) for x in cf_args])
            ea.append(ctypes.addressof(res_buffer),1)

    return blaze.Array(res_dd)

class BlazeExecutor(object):
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

    def run_append(self, dd):
        f = self.cfunc
        arg_s = self.arg_structs
        r = self.readers
        with dd.element_appender() as dst:
            for element in it_product(*[xrange(x) for x in self.outer_dims]):
                for struct, reader in izip(arg_s[:-1], r):
                    struct.e0 = ctypes.cast(reader.read_single(element),
                                            ctypes.POINTER(struct.e0._type_))

                with dst.buffered_ptr() as buf:
                    arg_s[-1].e0 = ctypes.cast(buf,
                                               ctypes.POINTER(arg_s[-1].e0._type_))
                    f(*arg_s)

    def run_write(self, dd):
        f = self.cfunc
        arg_s = self.arg_structs
        r = self.readers
        dst = dd.element_writer(len(self.outer_dims))
        for element in it_product(*[xrange(x) for x in self.outer_dims]):
            for struct, reader in izip(arg_s[:-1], r):
                struct.e0 = ctypes.cast(reader.read_single(element),
                                        ctypes.POINTER(struct.e0._type_))
            with dst.buffered_ptr(element) as buf:
                arg_s[-1].e0 = ctypes.cast(buf,
                                           ctypes.POINTER(arg_s[-1].e0._type_))
                f(*arg_s)

def execute_datadescriptor_ooc_2(dd, res_name=None):
    res_ds = dd.dshape
    res_shape, res_dt = to_numpy(dd.dshape)

    lifted = dd.kerneltree._fused.kernel.lift(1,'C')
    cf = lifted.ctypes_func
    res_ctype = cf.argtypes[-1]._type_
    args = [(ct._type_,
             arr.arr._data.element_reader(1),
             arr.arr.dshape.shape[1:])
            for ct, arr in izip(cf.argtypes[:-1], dd.args)]

    res_dd = BLZDataDescriptor(blz.zeros((0,) + res_shape[1:],
                                         dtype = res_dt,
                                         rootdir = res_name))

    with res_dd.element_appender() as dst:
        for i in xrange(res_shape[0]):
            # advance sources
            tpl = (i,)
            cf_args = [_mk_array_c_ref(t, er.read_single(tpl), sh)
                       for t, er, sh in args ]
            with dst.buffered_ptr() as dst_ptr:
                cf_args.append(_mk_array_c_ref(res_ctype,
                                               dst_ptr,
                                               res_shape[1:]))
                cf(*cf_args)

    return blaze.Array(res_dd)

res = execute_datadescriptor_ooc_2(d._data, 'foo.blz')
banner("result (ooc_2)")
print(res)
blaze.drop(blaze.Persist('foo.blz'))
del(res)

res = execute_datadescriptor_ooc(d._data, 'bar.blz')
banner("result (ooc)")
print(res)
blaze.drop(blaze.Persist('bar.blz'))
del(res)

banner("Executor iterating")
ex = BlazeExecutor(d._data, 1)
shape, dtype = to_numpy(d._data.dshape)
res_dd = BLZDataDescriptor(blz.zeros((0,)+shape[1:],
                                     dtype=dtype,
                                     rootdir='baz.blz'))


ex.run_append(res_dd)
res = blaze.Array(res_dd)
print(res)
blaze.drop(blaze.Persist('baz.blz'))
del res
del res_dd
del ex

banner("Executor memory no iter")
ex = BlazeExecutor(d._data, 0)
shape, dtype = to_numpy(d._data.dshape)
res_dd = NumPyDataDescriptor(np.empty(shape, dtype=dtype))

ex.run_write(res_dd)
res = blaze.Array(res_dd)
print(res)
del res
del res_dd
del ex

banner("Executor memory 1 iter")
ex = BlazeExecutor(d._data, 1)
shape, dtype = to_numpy(d._data.dshape)
res_dd = NumPyDataDescriptor(np.empty(shape, dtype=dtype))

ex.run_write(res_dd)
res = blaze.Array(res_dd)
print(res)
del res
del res_dd
del ex

dims = (1024*256, 512)

a = blaze.array(np.arange(np.prod(dims), dtype=np.float64).reshape(dims))
b = blaze.array(np.arange(np.prod(dims), dtype=np.float64).reshape(dims))

c = add(a, b)
d = mul(c, c)


print(d._data.kerneltree.ctypes_func)

def test_dims(d,outer):
    banner("Executer mem %d iters" % outer)
    ex =BlazeExecutor(d._data, outer)
    shape, dtype = to_numpy(d._data.dshape)
    res_dd = NumPyDataDescriptor(np.empty(shape, dtype=dtype))
    ex.run_write(res_dd)
    res = blaze.Array(res_dd)
    print (res)

#for i in xrange(len(dims)-1):
#    test_dims(d,i)


src_dd = d._data
sh, dt = to_numpy(src_dd.dshape)


banner('Executive interface (write interface)')
dst_dd = NumPyDataDescriptor(np.empty(sh, dtype=dt))
simple_execute_write(src_dd, dst_dd)
print(blaze.Array(dst_dd))
del dst_dd

banner('Executive interface (append interface)')
dst_dd = BLZDataDescriptor(blz.zeros((0,)+sh[1:],
                                    dtype=dt,
                                    rootdir='exec_test.blz'))


simple_execute_append(src_dd,dst_dd)
print(blaze.Array(dst_dd))
blaze.drop(blaze.Persist('exec_test.blz'))
del(dst_dd)

