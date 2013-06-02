from __future__ import absolute_import

# Implements the blaze.eval function

from .array import Array
from .constructors import empty
from .datadescriptor import (IDataDescriptor,
                BlazeFuncDescriptor, execute_expr_single)

def eval(arr, caps={'efficient-write': True}):
    """Evaluates a deferred blaze kernel tree
    data descriptor into a concrete array.
    If the array is already concrete, merely
    returns it unchanged.
    """
    if not isinstance(arr._data, BlazeFuncDescriptor):
        return arr
    result = empty(arr.dshape, caps)
    args = [arg.arr._data for arg in arr._data.args]
    kt = arr._data.kerneltree
    execute_expr_single(result._data, args,
                    kt.kernel.dshapes[-1],
                    kt.kernel.dshapes[:-1],
                    kt.single_ckernel)
    return result
