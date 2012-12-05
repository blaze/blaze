from functools import wraps
from threading import local
from thread import allocate_lock
from blaze.expr.paterm import matches

from ctypes import CFUNCTYPE

#------------------------------------------------------------------------
# Globals
#------------------------------------------------------------------------

_dispatch = local()
runtime_frozen = allocate_lock()

class NoDispatch(Exception):
    def __init__(self, aterm):
        self.aterm = aterm
    def __str__(self):
        return "No implementation for '%r'" % self.aterm

# WARNING, this is mutable
class Dispatcher(object):

    def __init__(self):
        self.funs  = {}
        self.costs = {}

    def install(self, matcher, fn, cost):
        self.funs[fn] = matcher
        self.costs[fn] = cost

    def dispatch(self, aterm):
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
    # directly
    with runtime_frozen:
        _dispatch.dispatcher.install(matcher, fn, cost)

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
