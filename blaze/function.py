# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys

class Dispatcher(object):
    """Dispatcher for overloaded functions"""
    def __init__(self):
        self.overloads = {}

    def add_overload(self, f, signature, kwds):
        # TODO: assert signature is compatible with current signatures
        self.overloads[f] = (signature, kwds)

    def dispatch(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        f = iter(self.overloads).next()
        return '<%s: %s>' % (f.__name__, list(self.overloads.itervalues()))

def overload(signature, func=None, **kwds):
    """
    Overload `func` with new signature, or find this function in the local
    scope with the same name.

        @overload('Array[dtype, ndim] -> dtype')
        def myfunc(...):
            ...
    """
    def decorator(f):
        dispatcher = func or sys._getframe(1).f_locals.get(f.__name__)
        dispatcher = dispatcher or Dispatcher()
        dispatcher.add_overload(f, signature, kwds)
        return dispatcher

    return decorator

def overloadable(f):
    """
    Make a function overloadable, useful if there's no useful defaults to
    overload on
    """
    return Dispatcher()

def best_match(overloads, typing_context, partial_solution):
    raise NotImplementedError