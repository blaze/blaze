"""

The RTS dispatcher. The logic that determines the most suitable
execution backend for a given function over ATerm operands.

    - Numba ( canonical )
    - numexpr
    - numpy
    - libc
    - Pandas?

    ...

    - User defined Python functions with datashape annotations
      and/or inferred.

The Numba backend is special in that the dispatcher will then feed
*the same aterm* down it is matching on, into Numba code generator to
actually build the kernel to execute.

The rest of the backends may requires some aterm munging to get
them into a form that is executable. I.e. casting into NumPy ( if
possible ), converting to a Numexpr expression, etc.

"""
from functools import wraps
from threading import local
from thread import allocate_lock
from blaze.expr.paterm import matches
from blaze.error import NoDispatch

from ctypes import CFUNCTYPE

#------------------------------------------------------------------------
# Globals
#------------------------------------------------------------------------

# Make sure that this isn't a moving target!
_dispatch = local()
runtime_frozen = allocate_lock()

# WARNING, this is mutable
class Dispatcher(object):

    def __init__(self):
        self.funs  = {}
        self.costs = {}

    def install(self, matcher, fn, cost):
        self.funs[fn] = matcher
        self.costs[fn] = cost

    def lookup(self, aterm):
        # canidate functions, functions matching the signature of
        # the term
        c = [f for f, sig in self.funs.iteritems() if matches(sig, aterm)]

        if len(c) == 0:
            raise NoDispatch(aterm)

        # the canidate which has the minimal cost function
        costs = [(f, self.costs[f](aterm)) for f in c]

        return min(costs, key=lambda x: x[1])

_dispatch.dispatcher = Dispatcher()

#------------------------------------------------------------------------
# Installling Functions
#------------------------------------------------------------------------

def lift(signature, constraints, mayblock=True):
    """ Lift a Python callable into Blaze with the given
    signature """
    def outer(pyfn):
        return PythonF(signature, pyfn, mayblock)
    return outer

def install(matcher, fn, cost):
    """ Install a function in the Blaze runtime, specializes
    based on the matcher. Assign the cost function to the
    selection of the function."""

    # This is the forward facing interface to installing
    # functions, users should never be accessing the dispatcher
    # directly since it's mutable and ugly...
    with runtime_frozen:
        _dispatch.dispatcher.install(matcher, fn, cost)

def lookup(aterm):
    return _dispatch.dispatcher.lookup(aterm)

#------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------

class PythonF(object):
    gil = True

    def __init__(self, signature, fn, mayblock):
        self.fn = fn
        self.mayblock = mayblock
        self.signature = signature

    @property
    def ptr(self):
        raise NotImplementedError

class ForeignF(object):

    def __init__(self, signature, ptr, gil, mayblock):
        self.ptr = ptr
        self.gil = gil
        self.mayblock = mayblock
        self.signature = signature
