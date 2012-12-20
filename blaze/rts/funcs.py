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


from thread import allocate_lock
from blaze.error import NoDispatch
from blaze.aterm import parse, match
from blaze.datashape.coretypes import dynamic

from blaze.expr.graph import Fun

from functools import wraps
from threading import local
from blaze.error import InvalidLibraryDefinton
from inspect import getargspec
from ctypes import CFUNCTYPE

#------------------------------------------------------------------------
# Globals
#------------------------------------------------------------------------

# Make sure that this isn't a moving target!
_dispatch = local()
runtime_frozen = allocate_lock()

# WARNING, this is mutable
class Dispatcher(object):
    """
    The dispatcher is a global object which holds the patterns
    for all functions known to Blaze, it runs through them all
    and determines which function

        a) match a given expression
        b) minimizes the cost of execution

    """

    def __init__(self):
        self.funs  = {}
        self.costs = {}

    def install(self, matcher, fn, cost):
        self.funs[fn] = matcher
        self.costs[fn] = cost

    def lookup(self, aterm):
        # canidate functions, functions matching the signature of
        # the term

        # Use the C libraries better pattern matching library,
        # find a function in the Blaze library that matches the
        # given aterm

        matched = []
        for f, sig in self.funs.iteritems():
            ismatch, _ = match(sig, str(aterm))
            if ismatch:
                matched.append(f)

        if len(matched) == 0:
            raise NoDispatch(aterm)

        # the canidate which has the minimal cost function
        costs = [(f, self.costs[f](aterm)) for f in matched]

        return min(costs, key=lambda x: x[1])

_dispatch.dispatcher = Dispatcher()

#------------------------------------------------------------------------
# Installling Functions
#------------------------------------------------------------------------

zerocost = lambda term: 0

def lift(signature, typesig, constraints=None, **params):
    """ Lift a Python callable into Blaze with the given
    signature. Splice a function graph node constructor in its
    place.

    This will ttransparently let the user lift functions in
    to the runtime and construct lazy graphs with what looks like
    immediate functions.
    """

    def outer(pyfn):
        assert callable(pyfn), "Lifted function must be callable"
        fname = pyfn.func_name

        try:
            parse(signature)
        except ATermSyntaxError as e:
            raise InvalidLibraryDefinton(*e.args + (fname,))

        # #effectful
        libcall = PythonFn(signature, pyfn)
        install(signature, libcall)

        sig = getargspec(pyfn)
        nargs = len(sig.args)

        # Return a new Fun() class that is a graph node
        # constructor.

        # Either we have a codomain annotation or we default to
        # the dynamic type.
        cod = params.pop('cod', dynamic)

        return type(pyfn.func_name, (Fun,), {
            'nargs'       : nargs,
            'fn'          : pyfn,
            'fname'       : fname,
            'typesig'     : typesig,
            'cod'         : cod,
            'constraints' : constraints,
        })

    return outer

def install(matcher, fn, cost=None):
    """ Install a function in the Blaze runtime, specializes
    based on the matcher. Assign the cost function to the
    selection of the function."""

    # This is the forward facing interface to installing
    # functions, users should never be accessing the dispatcher
    # directly since it's mutable and ugly...
    with runtime_frozen:
        costfn = cost or zerocost
        _dispatch.dispatcher.install(matcher, fn, costfn)

def lookup(aterm):
    return _dispatch.dispatcher.lookup(aterm)

#------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------

# A function in the Python interpreter
class PythonFn(object):
    gil = True

    def __init__(self, signature, fn, mayblock=False):
        self.fn = fn
        self.mayblock = mayblock
        self.signature = signature

    @property
    def ptr(self):
        raise NotImplementedError

# A function external to Python interpreter
class ExternalFn(object):

    def __init__(self, signature, ptr, gil, mayblock):
        self.ptr = ptr
        self.gil = gil
        self.mayblock = mayblock
        self.signature = signature
