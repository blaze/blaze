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
                             DeferredDescriptor)
from .py2help import reduce
from .datashape import to_numpy
from .air import prepare, interps
from .executive import simple_execute_append
from . import blz

#------------------------------------------------------------------------
# Execution Strategy
#------------------------------------------------------------------------

_eval_strategy = threading.local()
default_strategy = 'jit'

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
    strategy = strategy or current_strategy()

    if not arr._data.deferred:
        # TODO: This isn't right if the storage is different, requires
        #       a copy then.
        result = arr
    elif isinstance(arr._data, DeferredDescriptor):
        result = eval_deferred(arr, storage, caps, out, strategy)
    else:
        raise TypeError(("unexpected input to eval, "
                    "data desc has type %r") % type(arr._data))

    return result

def eval_deferred(arr, storage, caps, out, strategy):
    expr = arr._data.expr
    graph, ctx = expr

    # Construct and transform AIR
    func, env = prepare(expr, strategy)

    # Find evaluator
    interp = interps.lookup_interp(strategy)

    # Interpreter-specific compilation/assembly
    func, env = interp.compile(func, env)

    # Run with collected 'params' from the expression
    args = [ctx.terms[param] for param in ctx.params]
    result = interp.interpret(func, env, args=args, storage=storage,
                              caps=caps, out=out, strategy=strategy)

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

