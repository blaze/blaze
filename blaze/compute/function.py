"""
The purpose of this module is to create blaze functions. A Blaze Function
carries a polymorphic signature which allows it to verify well-typedness over
the input arguments, and to infer the result of the operation.

Blaze function also create a deferred expression graph when executed over
operands. A blaze function carries *implementations* that ultimately perform
the work. Implementations are indicated through the 'impl' keyword argument,
and may include:

    'py'    : Pure python implementation
    'numba' : Compiled numba function or compilable numba function
    'llvm'  : LLVM-compiled implementation
    'ctypes': A ctypes function pointer

Or a tuple for a combination of the above.
"""

from __future__ import print_function, division, absolute_import

import string
import textwrap
from itertools import chain

# TODO: Remove circular dependency between blaze.objects.Array and blaze.compute
import blaze
from ..py2help import dict_iteritems, exec_
from datashape import coretypes as T, dshape

from datashape.overloading import (overload, Dispatcher, match_by_weight,
                                   best_match, lookup_previous)
from ..datadescriptor import DeferredDescriptor
from .expr import construct, merge
from .strategy import PY, JIT

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def optional_decorator(f, continuation, args, kwargs):
    def decorator(f):
        return continuation(f, *args, **kwargs)

    if f is None:
        return decorator
    else:
        return decorator(f)


def blaze_args(args, kwargs):
    """Build blaze arrays from inputs to a blaze kernel"""
    args = [blaze.array(a) for a in args]
    kwargs = dict((v, blaze.array(k)) for k, v in dict_iteritems(kwargs))
    return args, kwargs


def collect_contexts(args):
    for term in args:
        if isinstance(term, blaze.Array) and term.expr:
            t, ctx = term.expr
            yield ctx


#------------------------------------------------------------------------
# Decorators
#------------------------------------------------------------------------
def function(signature, impl='python', **metadata):
    """
    Define an overload for a blaze function. Implementations may be associated
    by indicating a 'kind' through the `impl` argument.

    Parameters
    ----------
    signature: string or Type
        Optional function signature

    Usage
    -----

        @function
        def add(a, b):
            return a + b

    or

        @function('A -> A -> A') # All types are unified
        def add(a, b):
            return a + b
    """
    def decorator(f):
        # Look up previous blaze function
        blaze_func = lookup_previous(f)
        if blaze_func is None:
            # No previous function, create new one
            blaze_func = BlazeFunc()

        for impl in impls:
            kernel(blaze_func, impl, f, signature, **metadata)

        # Metadata
        blaze_func.add_metadata(metadata)
        if blaze_func.get_metadata('elementwise') is None:
            blaze_func.add_metadata({'elementwise': False})

        return blaze_func

    signature = dshape(signature)
    impls = impl
    if not isinstance(impls, tuple):
        impls = (impls,)

    if not isinstance(signature, T.Mono):
        # @blaze_func
        # def f(...): ...
        f = signature
        signature = None
        return decorator(f)
    else:
        # @blaze_func('A -> A -> B')
        # def f(...): ...
        return decorator


def elementwise(*args, **kwds):
    """
    Define a blaze element-wise kernel.
    """
    return function(*args, elementwise=True, **kwds)


def jit_elementwise(*args, **kwds):
    """
    Define a blaze element-wise kernel that can be jitted with numba.

    Keyword argument `python` indicates whether this is also a valid
    pure-python function (default: True).
    """
    if kwds.get(PY, True):
        impl = (PY, JIT)
    else:
        impl = JIT
    return elementwise(*args, impl=impl)


def apply_function(blaze_func, *args, **kwargs):
    """
    Apply blaze kernel `kernel` to the given arguments.

    Returns: a Deferred node representation the delayed computation
    """
    # -------------------------------------------------
    # Merge input contexts

    args, kwargs = blaze_args(args, kwargs)
    ctxs = collect_contexts(chain(args, kwargs.values()))
    ctx = merge(ctxs)

    # -------------------------------------------------
    # Find match to overloaded function

    overload, args = blaze_func.dispatcher.lookup_dispatcher(args, kwargs,
                                                             ctx.constraints)

    # -------------------------------------------------
    # Construct graph

    term = construct(blaze_func, ctx, overload, args)
    desc = DeferredDescriptor(term.dshape, (term, ctx))

    # TODO: preserve `user` metadata
    return blaze.Array(desc)


#------------------------------------------------------------------------
# Implementations
#------------------------------------------------------------------------
def kernel(blaze_func, impl_kind, kernel, signature, **metadata):
    """
    Define a new kernel implementation.
    """
    # Get dispatcher for implementation
    if isinstance(blaze_func, BlazeFunc):
        dispatcher = blaze_func.get_dispatcher(impl_kind)
    else:
        raise TypeError(
            "%s in current scope is not overloadable" % (blaze_func,))

    # Overload the right dispatcher
    overload(signature, dispatcher=dispatcher)(kernel)
    blaze_func.add_metadata(metadata, impl_kind=impl_kind)


def blaze_func(name, signature, **metadata):
    """
    Create a blaze function with the given signature. This is useful if there
    is not necessarily a python implementation available, or if we are
    generating blaze functions dynamically.
    """
    nargs = len(signature.argtypes)
    argnames = (string.ascii_lowercase + string.ascii_uppercase)[:nargs]
    source = textwrap.dedent("""
        def %(name)s(%(args)s):
            raise NotImplementedError("Python function for %(name)s")
    """ % {'name': name, 'args': ", ".join(argnames)})

    d = {}
    exec_(source, d, d)
    blaze_func = BlazeFunc()
    py_func = d[name]
    kernel(blaze_func, 'python', py_func, signature, **metadata)
    return blaze_func


def blaze_func_from_nargs(name, nargs, **metadata):
    """
    Create a blaze function with a given number of arguments.

    All arguments will be unified into a simple array type, which is also
    the return type, i.e. each argument is typed 'axes..., dtype', as well
    as the return type.

    This is to provide sensible typing for the dummy "reference" implementation.
    """
    argtype = dshape("axes..., dtype")
    signature = T.Function([argtype] * (nargs + 1))
    return blaze_func(name, signature, **metadata)


class BlazeFunc(object):
    """
    Blaze function. This is like the numpy ufunc object, in that it
    holds all the overloaded implementations of a function, and provides
    dispatch when called as a function. Objects of this type can be
    created directly, or using one of the decorators like @kernel
    or @elementwise.

    Attributes
    ----------
    dispatcher: Dispatcher
        Used to find the right overload

    metadata: { str : object }
        Additional metadata that may be interpreted by a Blaze AIR interpreter
    """

    def __init__(self, name=None):
        self.dispatchers = {}
        self.metadata = {}
        self.argnames = None

    @property
    def py_func(self):
        """Return the first python function that was subsequently overloaded"""
        return self.dispatcher.f

    @property
    def name(self):
        """Return the name of the blazefunc."""
        return self.dispatcher.f.__name__

    @property
    def __name__(self):
        """Return the name of the blazefunc."""
        return self.dispatcher.f.__name__

    @property
    def dispatcher(self):
        """Default dispatcher that define blaze semantics (pure python)"""
        return self.dispatchers[PY]

    @property
    def available_strategies(self):
        return list(self.dispatchers)

    def get_dispatcher(self, impl_kind):
        """Get the overloaded dispatcher for the given implementation kind"""
        if impl_kind not in self.dispatchers:
            self.dispatchers[impl_kind] = Dispatcher()
        return self.dispatchers[impl_kind]

    def matches(self, impl_kind, argtypes, constraints=None):
        """
        Find all matching overloads for a given implementation kind and
        argument types.
        """
        return match_by_weight(self.get_dispatcher(impl_kind), argtypes,
                               constraints=constraints)

    def best_match(self, impl_kind, argtypes, constraints=None):
        """
        Find the best implementation of `impl_kind` using `argtypes`.
        """
        return best_match(self.get_dispatcher(impl_kind), argtypes,
                          constraints=constraints)

    def add_metadata(self, md, impl_kind=PY):
        """
        Associate metadata with an overloaded implementation.
        """
        if impl_kind not in self.metadata:
            self.metadata[impl_kind] = {}

        metadata = self.metadata[impl_kind]

        # Verify compatibility
        for k in md:
            if k in metadata:
                assert metadata[k] == md[k], (metadata[k], md[k])
        # Update
        metadata.update(md)

    def get_metadata(self, key, impl_kind=PY):
        return self.metadata[impl_kind].get(key)

    __call__ = apply_function

    def __str__(self):
        arg = self.py_func
        if arg is None:
            arg = "<empty>"
        else:
            arg = ".".join([arg.__module__, arg.__name__])
        return "BlazeFunc(%s)" % (arg,)
