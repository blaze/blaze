# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import inspect
from collections import namedtuple, defaultdict
from itertools import chain

from blaze import error
from blaze.util import flatargs, listify, alpha_equivalent
from blaze.datashape import (coretypes as T, unify, dshape,
                             dummy_signature)

class Dispatcher(object):
    """Dispatcher for overloaded functions"""

    def __init__(self):
        self.f = None
        self.overloads = []
        self.argspec = None

    def add_overload(self, f, signature, kwds, argspec=None):
        # TODO: assert signature is "compatible" with current signatures
        if self.f is None:
            self.f = f

        # Process signature
        argspec = argspec or inspect.getargspec(f)
        if self.argspec is None:
            self.argspec = argspec
        alpha_equivalent(self.argspec, argspec)

        # TODO: match signature to be a Function type with correct arity
        self.overloads.append((f, signature, kwds))

    def lookup_dispatcher(self, args, kwargs, constraints=None):
        assert self.f is not None
        args = flatargs(self.f, tuple(args), kwargs)
        types = list(map(T.typeof, args))
        match = best_match(self, types, constraints)
        return match, args

    def dispatch(self, *args, **kwargs):
        match, args = self.lookup_dispatcher(args, kwargs)
        return match.func(*args)

    def simple_dispatch(self, *args, **kwargs):
        assert self.f is not None
        args = flatargs(self.f, args, kwargs)
        types = list(map(T.typeof, args))
        candidates = find_matches(self.overloads, types)
        if len(candidates) != 1:
            raise error.OverloadError(
                "Cannot perform simple dispatch with %d input types")

        [dst_sig, sig, func] = candidates
        # TODO: convert argument types using dst_sig
        return func(*args)

    __call__ = dispatch

    def __repr__(self):
        signatures = [sig for f, sig, _ in self.overloads]
        return '<%s: \n%s>' % (self.f and self.f.__name__,
                               "\n".join("    %s" % (s,) for s in signatures))

def overload(signature, func=None, **kwds):
    """
    Overload `func` with new signature, or find this function in the local
    scope with the same name.

        @overload('Array[dtype, ndim] -> dtype')
        def myfunc(...):
            ...
    """
    def decorator(f, signature=signature):
        if signature is None:
            signature = dummy_signature(f)
        else:
            signature = dshape(signature)

        dispatcher = func or f.__globals__.get(f.__name__)
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

Overload = namedtuple('Overload', 'resolved_sig, sig, func, constraints, kwds')

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

    Returns
    -------
    Overloaded function as an `Overload` instance.
    """
    from blaze.datashape import coercion_cost
    overloads = func.overloads

    # -------------------------------------------------
    # Find candidates

    candidates = find_matches(overloads, argtypes, constraints or [])

    # -------------------------------------------------
    # Weigh candidates

    matches = defaultdict(list)
    for match in candidates:
        params = match.sig.parameters[:-1]
        try:
            weight = sum([coercion_cost(a, p) for a, p in zip(argtypes, params)])
        except error.CoercionError:
            pass
        else:
            matches[weight].append(match)

    if not matches:
        raise error.OverloadError(
            "No overload for function %s matches for argtypes (%s)" % (
                                    func, ", ".join(map(str, argtypes))))

    # -------------------------------------------------
    # Return candidate with minimum weight

    candidates = matches[min(matches)]
    if len(candidates) > 1:
        #print(matches)
        raise error.OverloadError(
            "Ambiguous overload for function %s with inputs (%s): \n%s" % (
                func, ", ".join(map(str, argtypes)),
                "\n".join("    %s" % (overload.resolved_sig,) for overload in candidates)))
    else:
        return candidates[0]

@listify
def find_matches(overloads, argtypes, constraints=()):
    """Find all overloads that unify with the given inputs"""
    input = T.Function(*argtypes + [T.TypeVar('R')])
    for func, sig, kwds in overloads:
        assert isinstance(sig, T.Function), sig

        # -------------------------------------------------
        # Error checking
        l1, l2 = len(sig.argtypes), len(argtypes)
        if l1 != l2:
            raise TypeError(
                "Expected %d args, got %d for function %s" % (l1, l2, func))

        # -------------------------------------------------
        # Unification

        equations = list(chain([(input, sig)], constraints))
        broadcasting = [True] * l1

        try:
            result, remaining = unify(equations, broadcasting)
        except error.UnificationError:
            continue
        else:
            dst_sig = result[0]
            yield Overload(dst_sig, sig, func, remaining, kwds)
