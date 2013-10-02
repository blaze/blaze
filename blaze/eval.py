from __future__ import absolute_import

# Implements the blaze.eval function

import threading
from contextlib import contextmanager

import blaze
from .array import Array
from .constructors import empty
from .datadescriptor import (IDataDescriptor,
                             BlazeFuncDeprecatedDescriptor,
                             BLZDataDescriptor,
                             DeferredDescriptor,
                             execute_expr_single)
from .py2help import reduce
from .datashape import to_numpy
from .air import prepare, interps
from .executive import simple_execute_append
from . import blz

#------------------------------------------------------------------------
# Execution Strategy
#------------------------------------------------------------------------

_eval_strategy = threading.local()
default_strategy = 'py'

@contextmanager
def strategy(strategy):
    """
    Set the evaluation strategy for expressions evaluating in this thread.

    Parameters
    ----------
    strategy: str
        Evaluation strategy. Currently supported:

            * 'py'      Evaluation using Python and possibly operations of
                        underlying containers
            * 'eval'    Try to assemble a more efficient evaluation kernel
                        that performs fusion as much as possible
            * 'jit'     JIT-compile the expression to machine code specialized
                        entirely to the expression

        The above strategies are listed in order of fast- to slow-assembly,
        and from slow evaluation to fast evaluation.
    """
    old = current_strategy()
    set_strategy(strategy)
    yield
    set_strategy(old)

def set_strategy(strategy):
    _eval_strategy.strategy = strategy

def current_strategy():
    """Return the current evaluation strategy"""
    try:
        return _eval_strategy.strategy
    except AttributeError:
        return default_strategy

#------------------------------------------------------------------------
# Eval
#------------------------------------------------------------------------

def eval(arr, storage=None, caps={'efficient-write': True}, out=None,
         strategy=None):
    """Evaluates a deferred blaze kernel tree
    data descriptor into a concrete array.
    If the array is already concrete, merely
    returns it unchanged.

    Parameters
    ----------
    storage:
        Where to store the result
        TODO: elaborate on values and input type

    caps: { str : object }
        Capabilities for evaluation and storage
        TODO: elaborate on values

    out: Array
        Output array to store the result in, or None for a new array

    strategy: str
        Evaluation strategy.
        Currently supported: 'py', 'eval',
    """
    strategy = strategy or current_strategy()

    if not arr._data.deferred:
        result = arr
    elif isinstance(arr._data, DeferredDescriptor):
        result = eval_deferred(arr, storage, caps, out, strategy)
    else:
        kt = arr._data.kerneltree.fuse()
        if storage is not None:
            result = eval_blz(arr, storage, caps, out, strategy)
        else: # in memory path
            result = eval_ckernel(arr, storage, caps, out, strategy, kt)

        for name in ['axes', 'user', 'labels']:
            setattr(result, name, getattr(arr, name))

    return result

def eval_deferred(arr, storage, caps, out, strategy):
    expr = arr._data.expr
    graph, ctx = expr

    # Construct and transform AIR
    func, env = prepare(expr, strategy)

    # Find evaluator
    interp = interps.lookup_interp(strategy)

    # Interpreter-specific compilation/assembly
    func = interp.compile(func, env)

    # Run with collected 'params' from the expression
    result = interp.run(func, args=[ctx.terms[param] for param in ctx.params],
                        storage=storage, caps=caps, out=out, strategy=strategy)

    return result

def eval_blz(arr, storage, caps, out, strategy):
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

    return result

def eval_ckernel(arr, storage, caps, out, strategy, kt):
    result = empty(arr.dshape, caps)
    args = [arg.arr._data for arg in arr._data.args]
    ubck = kt.make_unbound_ckernel(strided=False)
    ck = ubck.bind(result._data, args)
    execute_expr_single(result._data, args,
                        kt.kernel.dshapes[-1],
                        kt.kernel.dshapes[:-1],
                        ck)

    return result

#------------------------------------------------------------------------
# Append
#------------------------------------------------------------------------

def append(arr, values):
    """Append a list of values."""
    # XXX If not efficient appends supported, this should raise
    # a `PerformanceWarning`
    if hasattr(arr._data, 'append'):
        arr._data.append(values)
    else:
        raise NotImplementedError('append is not implemented for this '
                                  'object')

