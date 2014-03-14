"""
The purpose of this module is to create blaze functions. A Blaze Function
carries a polymorphic signature which allows it to verify well-typedness over
the input arguments, and to infer the result of the operation.

Blaze function also create a deferred expression graph when executed over
operands. A blaze function carries *implementations* that ultimately perform
the work. Implementations are indicated through the 'impl' keyword argument,
and may include:

    'ckernel'    : Blaze ckernel-based function.
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

from datashape.overloading import overload, Dispatcher
from datashape.overload_resolver import OverloadResolver
from ..datadescriptor import DeferredDescriptor
from .expr import construct, merge_contexts
from .strategy import CKERNEL

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def collect_contexts(args):
    for term in args:
        if isinstance(term, blaze.Array) and term.expr:
            t, ctx = term.expr
            yield ctx


def lookup_previous(f, scopes=None):
    """
    Lookup a previous function definition in the current namespace, i.e.
    for overloading purposes.
    """
    if scopes is None:
        scopes = []

    scopes.append(f.__globals__)

    for scope in scopes:
        if scope.get(f.__name__):
            return scope[f.__name__]

    return None

#------------------------------------------------------------------------
# Decorators
#------------------------------------------------------------------------
def function(signature, impl=CKERNEL, **metadata):
    """
    Define an overload for a blaze function. Implementations may be associated
    by indicating a 'kind' through the `impl` argument.

    Parameters
    ----------
    signature: string or Type
        Optional function signature

    Usage
    -----

        @function('(A, A) -> A') # All types are promoted
        def add(a, b):
            return a + b
    """
    def decorator(f):
        # Look up previous blaze function
        bf = lookup_previous(f)
        if bf is None:
            # No previous function, create new one
            bf = BlazeFunc(f.__module__, f.__name__)

        for impl in impls:
            kernel(bf, impl, f, signature, **metadata)

        # Metadata
        bf.add_metadata(metadata)
        if bf.get_metadata('elementwise') is None:
            bf.add_metadata({'elementwise': False})

        return bf

    signature = dshape(signature)
    if isinstance(signature, T.DataShape) and len(signature) == 1:
        signature = signature[0]
    impls = impl
    if not isinstance(impls, tuple):
        impls = (impls,)

    if not isinstance(signature, T.Function):
        raise TypeError('Blaze @function decorator requires a function signature')
    else:
        # @function('(A, A) -> B')
        # def f(...): ...
        return decorator


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


class BlazeFunc(object):
    """
    Blaze function. This is like the numpy ufunc object, in that it
    holds all the overloaded implementations of a function, and provides
    dispatch when called as a function. Objects of this type can be
    created directly, or using one of the decorators like @function .

    Attributes
    ----------
    ores: OverloadResolver
        Used to find the right overload

    metadata: { str : object }
        Additional metadata that may be interpreted by a Blaze AIR interpreter
    """

    def __init__(self, module, name):
        self._module = module
        self._name = name
        self.metadata = {}
        self.dispatchers = {CKERNEL: Dispatcher()}

    @property
    def name(self):
        """Return the name of the blazefunc."""
        return self._name

    @property
    def module(self):
        return self._module

    @property
    def fullname(self):
        return self._module + '.' + self._name

    @property
    def available_strategies(self):
        return list(self.dispatchers)

    @property
    def dispatcher(self):
        return self.dispatchers[CKERNEL]

    def get_dispatcher(self, impl_kind):
        """Get the overloaded dispatcher for the given implementation kind"""
        if impl_kind not in self.dispatchers:
            self.dispatchers[impl_kind] = Dispatcher()
        return self.dispatchers[impl_kind]

    def best_match(self, impl_kind, argtypes):
        """
        Find the best implementation of `impl_kind` using `argtypes`.
        """
        dispatcher = self.get_dispatcher(impl_kind)
        return dispatcher.best_match(argtypes)

    def add_metadata(self, md, impl_kind=CKERNEL):
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

    def get_metadata(self, key, impl_kind=CKERNEL):
        return self.metadata[impl_kind].get(key)

    def __call__(self, *args, **kwargs):
        """
        Apply blaze kernel `kernel` to the given arguments.

        Returns: a Deferred node representation the delayed computation
        """
        # -------------------------------------------------
        # Merge input contexts

        args = [blaze.array(a) for a in args]
        ctxs = collect_contexts(chain(args, kwargs.values()))
        ctx = merge_contexts(ctxs)

        # -------------------------------------------------
        # Find match to overloaded function

        overload, args = self.dispatcher.lookup_dispatcher(args, kwargs)

        # -------------------------------------------------
        # Construct graph

        term = construct(self, ctx, overload, args)
        desc = DeferredDescriptor(term.dshape, (term, ctx))

        # TODO: preserve `user` metadata
        return blaze.Array(desc)

    def __str__(self):
        return "BlazeFunc %s" % self.name

    def __repr__(self):
        # TODO proper repr
        return str(self)
