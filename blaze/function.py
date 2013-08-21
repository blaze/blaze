# -*- coding: utf-8 -*-

"""
The purpose of this module is to create blaze functions. A Blaze Function
carries a polymorphic signature which allows it to verify well-typedness over
the input arguments, and to infer the result of the operation.

A blaze function also creates a deferred expression graph.
"""

from __future__ import print_function, division, absolute_import
import functools

from .overloading import overload
from .deferred import Deferred
from .util import flatargs


def elementwise(signature):
    """
    Define an element-wise kernel.
    """
    def decorator(f):
        return overload(signature)(f, elementwise=True)

    if not isinstance(signature, basestring):
        # signature
        f = signature
        signature = None
        return decorator(f)

    return decorator

def deferred_wrapper(f):
    """
    Given a (typed) Blaze function, capture the arguments, type the result,
    create a new node in the expression graph and return a Deferred.
    """
    from .expr import construct

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        args = flatargs(args, kwargs)
        expr = construct.construct(f, *args)
        (term, context) = expr
        return Deferred(term.dshape, expr)
    return wrapper


def best_match(overloads, typing_context, partial_solution):
    raise NotImplementedError

