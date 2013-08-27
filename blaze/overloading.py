# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys
import collections
from pprint import pformat

from blaze import error
from blaze.util import listify
from blaze.datashape import coretypes as T, unify

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

#------------------------------------------------------------------------
# Matching
#------------------------------------------------------------------------

def best_match(func, argtypes, constraints=None):
    """
    Find a best match in for overloaded function `func` given `argtypes`.

    Parameters
    ----------
    func: Dispatcher
        Overloaded Blaze function

    argtypes: [Mono]
        List of input argument types

    constraints: [(TypeVar, Mono)]
        Optional set of constraints, see unification.py
    """
    from blaze.datashape import coerce
    overloads = func.overloads

    # -------------------------------------------------
    # Find candidates

    candidates = find_matches(overloads, argtypes, constraints)
    input = T.Function(*argtypes + [T.TypeVar('R')])

    # -------------------------------------------------
    # Weigh candidates

    matches = collections.defaultdict(list)
    for candidate in candidates:
        dst_sig, sig, func = candidate
        try:
            weight = coerce(input, dst_sig)
        except error.CoercionError:
            pass
        else:
            matches[weight].append(candidate)

    if not matches:
        raise error.OverloadError(
            "No overload for function %s matches input %s" % (func, input))

    # -------------------------------------------------
    # Return candidate with minimum weight

    candidates = matches[min(matches)]
    if len(candidates) > 1:
        raise error.OverloadError(
            "Ambiguous overload for function %s with input %s: %s" % (
                func, input, pformat(candidates)))
    else:
        return candidates[0]

@listify
def find_matches(overloads, argtypes, constraints=None):
    """Find all overloads that unify with the given inputs"""
    input = T.Function(*argtypes + [T.TypeVar('R')])
    for sig, func in overloads:
        assert isinstance(sig, T.Function), sig

        # -------------------------------------------------
        # Error checking
        l1, l2 = len(sig.argtypes), len(argtypes)
        if l1 != l2:
            raise TypeError(
                "Expected %d args, got %d for function %s" % (l1, l2, func))

        # -------------------------------------------------
        # Unification

        constraints = [(input, sig)] + (constraints or [])
        broadcasting = [True] * l1

        try:
            [dst_sig], _ = unify(constraints, broadcasting)
        except error.UnificationError:
            continue
        else:
            yield dst_sig, sig, func
