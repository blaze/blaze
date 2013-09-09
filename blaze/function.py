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

from .deferred import apply_kernel
from .overloading import overload, Dispatcher

from blaze.py2help import basestring

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def lookup_previous(f):
    """
    Lookup a previous function definition in the current namespace, i.e.
    for overloading purposes.
    """
    return f.func_globals.get(f.__name__)

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
        func = lookup_previous(f)
        if isinstance(func, Kernel):
            func = func.dispatcher
        elif isinstance(func, types.FunctionType):
            raise TypeError(
                "Function %s in current scope is not overloadable" % (func,))
        else:
            func = Dispatcher()

        dispatcher = overload(signature, func=func)(f)

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
    """
    Blaze kernel.

    Attributes
    ----------
    dispatcher: Dispatcher
        Used to find the right overload

    impls: { str : object }
        Overloaded implementations, mapping from implementation 'kind'
        (e.g. python, ctypes, llvm, etc) to an implementation

    metadata: { str : object }
        Additional metadata that may be interpreted by a Blaze AIR interpreter
    """

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

    __call__ = apply_kernel

    def __str__(self):
        arg = self.dispatcher.f
        if arg is None:
            arg = "<empty>"
        else:
            arg = ".".join([arg.__module__, arg.__name__])
        return "Kernel(%s)" % (arg,)