from __future__ import absolute_import

# Implements the blaze.eval function

import threading
from contextlib import contextmanager

from .array import Array
from .constructors import empty
from .datadescriptor import (IDataDescriptor,
                             BlazeFuncDescriptor,
                             BLZDataDescriptor,
                             execute_expr_single)
from .py2help import reduce
from .datashape import to_numpy
from .executive import simple_execute_append
from . import blz

#------------------------------------------------------------------------
# Execution Strategy
#------------------------------------------------------------------------

_eval_strategy = threading.local()
_eval_strategy.strategy = 'llvm'

@contextmanager
def strategy(strategy):
    old = _eval_strategy.strategy
    _eval_strategy.strategy = strategy
    yield
    _eval_strategy.strategy = old

def current_strategy():
    return _eval_strategy.strategy

#------------------------------------------------------------------------
# Eval
#------------------------------------------------------------------------

def eval(arr, storage=None, caps={'efficient-write': True}, out=None,
         strategy=None):
    """Evaluates a deferred blaze kernel tree
    data descriptor into a concrete array.
    If the array is already concrete, merely
    returns it unchanged.
    """
    if not arr._data.deferred:
        return arr

    if strategy is None:
        strategy = _eval_strategy.strategy

    kt = arr._data.kerneltree.fuse()
    if storage is not None:
        from operator import mul
        # out of core path
        res_dshape, res_dt = to_numpy(arr._data.dshape)
        dst_dd = BLZDataDescriptor(blz.zeros((0,)+res_dshape[1:], res_dt,
                                             rootdir=storage.path))

        # this is a simple heuristic for chunk size:
        row_size = res_dt.itemsize
        if len(res_dshape) > 1:
            row_size *= reduce(mul, res_dshape[1:])

        chunk_size = max(1, (1024*1024) // row_size)
        simple_execute_append(arr._data, dst_dd, chunk=chunk_size)
        result = Array(dst_dd)
    else: # in memory path
        result = empty(arr.dshape, caps)
        args = [arg.arr._data for arg in arr._data.args]
        ubck = kt.make_unbound_ckernel(strided=False)
        ck = ubck.bind(result._data, args)
        execute_expr_single(result._data, args,
                            kt.kernel.dshapes[-1],
                            kt.kernel.dshapes[:-1],
                            ck)

    for name in ['axes', 'user', 'labels']:
        setattr(result, name, getattr(arr, name))

    return result

def append(arr, values):
    """Append a list of values."""
    # XXX If not efficient appends supported, this should raise
    # a `PerformanceWarning`
    if hasattr(arr._data, 'append'):
        arr._data.append(values)
    else:
        raise NotImplementedError('append is not implemented for this '
                                  'object')

