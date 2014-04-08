from __future__ import absolute_import, division, print_function

"""Implements the blaze.eval function"""

from .air import compile, run
from .. import array

#------------------------------------------------------------------------
# Eval
#------------------------------------------------------------------------

def eval(arr, ddesc=None, caps={'efficient-write': True},
         out=None, debug=False):
    """Evaluates a deferred blaze kernel tree
    data descriptor into a concrete array.
    If the array is already concrete, merely
    returns it unchanged.

    Parameters
    ----------
    ddesc: DDesc instance, optional
        A data descriptor for storing the result, if evaluating to a BLZ
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
    if arr.ddesc.capabilities.deferred:
        result = eval_deferred(
            arr, ddesc=ddesc, caps=caps, out=out, debug=debug)
    elif arr.ddesc.capabilities.remote:
        # Retrieve the data to local memory
        # TODO: Caching should play a role here.
        result = array(arr.ddesc.dynd_arr())
    else:
        # TODO: This isn't right if the data descriptor is different, requires
        #       a copy then.
        result = arr

    return result


def eval_deferred(arr, ddesc, caps, out, debug=False):
    expr = arr.ddesc.expr
    graph, ctx = expr

    # collected 'params' from the expression
    args = [ctx.terms[param] for param in ctx.params]

    func, env = compile(expr, ddesc=ddesc)
    result = run(func, env, ddesc=ddesc, caps=caps, out=out, debug=debug)

    return result

#------------------------------------------------------------------------
# Append
#------------------------------------------------------------------------

def append(arr, values):
    """Append a list of values."""
    # XXX If not efficient appends supported, this should raise
    # a `PerformanceWarning`
    if arr.ddesc.capabilities.appendable:
        arr.ddesc.append(values)
    else:
        raise ValueError('Data source cannot be appended to')

