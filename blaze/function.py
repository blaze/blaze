# -*- coding: utf-8 -*-

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

import types
from itertools import chain

import blaze
from blaze.datashape import coretypes as T, dshape
from blaze.overloading import overload, Dispatcher
from blaze.datadescriptor import DeferredDescriptor
from blaze.expr.context import merge
from blaze.py2help import basestring, dict_iteritems

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

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

def kernel(signature, impl='python', **metadata):
    """
    Define an blaze python-level kernel. Further implementations may be
    associated with this overloaded kernel using the 'implement' method.

    Parameters
    ----------
    signature: string or Type
        Optional function signature

    Usage
    -----

        @kernel
        def add(a, b):
            return a + b

    or

        @kernel('a -> a -> a') # All types are unified
        def add(a, b):
            return a + b
    """
    def decorator(f):
        func = lookup_previous(f)
        if isinstance(func, BlazeFunc):
            func = func.dispatcher
        elif isinstance(func, types.FunctionType):
            raise TypeError(
                "Function %s in current scope is not overloadable" % (func,))
        else:
            func = Dispatcher()

        dispatcher = overload(signature, func=func)(f)

        if isinstance(f, types.FunctionType):
            kernel = BlazeFunc(dispatcher)
        else:
            assert isinstance(f, BlazeFunc), f
            kernel = f

        metadata.setdefault('elementwise', True)
        kernel.add_metadata(metadata)
        if impl != 'python':
            kernel.implement(f, signature, impl, f)
        return kernel

    signature = dshape(signature)

    if not isinstance(signature, T.Mono):
        # @kernel
        # def f(...): ...
        f = signature
        signature = None
        return decorator(f)
    else:
        # @kernel('a -> a -> b')
        # def f(...): ...
        return decorator

def elementwise(*args, **kwds):
    """
    Define a blaze element-wise kernel.
    """
    return kernel(*args, elementwise=True, **kwds)

def jit_elementwise(*args):
    """
    Define a blaze element-wise kernel that can be jitted with numba.
    """
    return elementwise(*args, impl='numba')

#------------------------------------------------------------------------
# Application
#------------------------------------------------------------------------

def apply_kernel(kernel, *args, **kwargs):
    """
    Apply blaze kernel `kernel` to the given arguments.

    Returns: a Deferred node representation the delayed computation
    """
    from .expr import construct

    # -------------------------------------------------
    # Merge input contexts

    args, kwargs = blaze_args(args, kwargs)
    ctxs = collect_contexts(chain(args, kwargs.values()))
    ctx = merge(ctxs)

    # -------------------------------------------------
    # Find match to overloaded function

    overload, args = kernel.dispatcher.lookup_dispatcher(args, kwargs,
                                                         ctx.constraints)

    # -------------------------------------------------
    # Construct graph

    term = construct.construct(kernel, ctx, overload, args)
    desc = DeferredDescriptor(term.dshape, (term, ctx))

    # TODO: preserve `user` metadata
    return blaze.Array(desc)

#------------------------------------------------------------------------
# Implementations
#------------------------------------------------------------------------

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

    impls: { (py_func, signature) : object }
        Overloaded implementations, mapping from implementation 'kind'
        (e.g. python, numba, ckernel, llvm, etc) to an implementation

    metadata: { str : object }
        Additional metadata that may be interpreted by a Blaze AIR interpreter
    """

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        self.impls = {}
        self.metadata = {}

    def implement(self, py_func, signature, impl_kind, kernel):
        """
        Add an implementation kernel for the overloaded function `py_func`.

        Arguments
        ---------
        py_func: FunctionType
            The original overloaded Python function.

            Note: Decorators like @overload etc do not return the original
                  Python function!

        signature: Type
            The function signature of the overload in question

        impl_kind: str
            Type of implementation, e.g. 'python', 'numba', 'ckernel', etc

        kernel: some implementation object
            Some object depending on impl_kind, which implements the
            overload.
        """
        impls_dict = self.impls.setdefault((py_func, str(signature)), {})
        impls_list = impls_dict.setdefault(impl_kind, [])
        impls_list.append(kernel)

    def find_impls(self, py_func, signature, impl_kind):
        return self.impls.get((py_func, str(signature)), {}).get(impl_kind)

    def add_metadata(self, md):
        # Verify compatibility
        for k in md:
            if k in self.metadata:
                assert self.metadata[k] == md[k], (self.metadata[k], md[k])
        # Update
        self.metadata.update(md)

    __call__ = apply_kernel

    def __str__(self):
        arg = self.dispatcher.f
        if arg is None:
            arg = "<empty>"
        else:
            arg = ".".join([arg.__module__, arg.__name__])
        return "BlazeFunc(%s)" % (arg,)
