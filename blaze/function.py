# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from .overloading import overload

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


def best_match(overloads, typing_context, partial_solution):
    raise NotImplementedError

