from __future__ import absolute_import

# Implements the blaze.eval function

from .array import Array
from .constructors import empty
from .datadescriptor import (IDataDescriptor,
                             BlazeFuncDescriptor,
                             BLZDataDescriptor,
                             execute_expr_single)

from .datashape import to_numpy
from .executive import simple_execute_append
from . import blz


def eval(arr, persist=None, caps={'efficient-write': True}):
    """Evaluates a deferred blaze kernel tree
    data descriptor into a concrete array.
    If the array is already concrete, merely
    returns it unchanged.
    """
    if not arr._data.deferred:
        return arr

    kt = arr._data.kerneltree
    if persist is not None:
        # out of core path
        res_dshape, res_dt = to_numpy(arr._data.dshape)
        kt = arr._data.kerneltree
        dst_dd = BLZDataDescriptor(blz.zeros((0,)+res_dshape[1:], res_dt,
                                             rootdir=persist.path))
        simple_execute_append(arr._data, dst_dd)
        result = Array(dst_dd)
    else:
        # in memory path
        result = empty(arr.dshape, caps)
        args = [arg.arr._data for arg in arr._data.args]
        ubck = kt.unbound_single_ckernel
        ck = ubck.bind(result._data, args)
        execute_expr_single(result._data, args,
                            kt.kernel.dshapes[-1],
                            kt.kernel.dshapes[:-1],
                            ck)
    return result
