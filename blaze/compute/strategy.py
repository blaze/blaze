"""
Blaze execution strategy.
"""

from __future__ import absolute_import, division, print_function

import threading
from contextlib import contextmanager

#------------------------------------------------------------------------
# Strategies
#------------------------------------------------------------------------

OOC     = 'out-of-core'
CKERNEL = 'ckernel'
JIT     = 'jit'
PY      = 'python'

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
