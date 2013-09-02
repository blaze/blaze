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
import inspect
import functools

from .overloading import overload
from .deferred import Deferred
from .util import flatargs

from blaze.py2help import basestring

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def lookup_func(f=None):
    return f or f.func_globals.get(f.__name__)

def optional_decorator(f, continuation, args, kwargs):
    def decorator(f):
        return continuation(f, *args, **kwargs)

    if f is None:
        return decorator
    else:
        return decorator(f)

#------------------------------------------------------------------------
# Decorators
#------------------------------------------------------------------------

def kernel(signature, **metadata):
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
        f = lookup_func(f)
        dispatcher = overload(signature, f)

        if isinstance(f, types.FunctionType):
            kernel = Kernel(dispatcher)
        else:
            assert isinstance(f, Kernel), f
            kernel = f

        kernel.add_metadata(metadata)
        return kernel

    if not isinstance(signature, basestring):
        # @kernel
        # def f(...): ...
        f = signature
        signature = None
        return decorator(f)
    else:
        # @kernel('a -> a -> b')
        # def f(...): ...
        return decorator

def elementwise(*args):
    """
    Define a blaze element-wise kernel.
    """
    return kernel(*args, elementwise=True)

#------------------------------------------------------------------------
# Implementations
#------------------------------------------------------------------------

class Kernel(object):
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        self.impls = {}
        self.metadata = {}

    def implement(self, signature, impl, func=None):
        return optional_decorator(func, self._implement, signature, impl)

    def _implement(self, func, signature, impl):
        self.impls[impl].append((signature, func))
        return func

    def add_metadata(self, md):
        # Verify compatibility
        for k in md:
            if k in self.metadata:
                assert self.metadata[k] == md[k], (self.metadata[k], md[k])
        # Update
        self.metadata.update(md)

    def __call__(self, *args, **kwargs):
        from .expr import construct

        func, dst_sig, args = self.dispatcher.lookup_dispatcher(args, kwargs)
        expr = construct.construct(func, dst_sig, *args)
        (term, context) = expr
        return Deferred(term.dshape, expr)
