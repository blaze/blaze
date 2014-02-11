from __future__ import absolute_import, division, print_function

# Implements the blaze.eval function

from .strategy import JIT, current_strategy
from .air import compile, run
from .. import array

#------------------------------------------------------------------------
# Eval
#------------------------------------------------------------------------

def eval(arr, storage=None, caps={'efficient-write': True}, out=None):
    """Evaluates a deferred blaze kernel tree
    data descriptor into a concrete array.
    If the array is already concrete, merely
    returns it unchanged.

    Parameters
    ----------
    storage: blaze.Storage, optional
        Where to store the result, if evaluating to a BLZ
        output or (in the future) to a distributed array.

    caps: { str : object }
        Capabilities for evaluation and storage
        TODO: elaborate on values

    out: Array
        Output array to store the result in, or None for a new array

    strategy: str
        Evaluation strategy.
        Currently supported: 'py', 'jit'
    """
    if arr._data.capabilities.deferred:
        result = eval_deferred(arr, storage=storage, caps=caps, out=out)
    elif arr._data.capabilities.remote:
        # Retrieve the data to local memory
        # TODO: Caching should play a role here.
        result = array(arr._data.dynd_arr())
    else:
        # TODO: This isn't right if the storage is different, requires
        #       a copy then.
        result = arr

    return result

def eval_deferred(arr, storage, caps, out):
    expr = arr._data.expr
    graph, ctx = expr

    # collected 'params' from the expression
    args = [ctx.terms[param] for param in ctx.params]

    func, env = compile(expr, storage=storage)
    result = run(func, env, storage=storage, caps=caps, out=out)
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

