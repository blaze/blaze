# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import string
import inspect
import functools
import collections
from functools import partial

#------------------------------------------------------------------------
# General purpose
#------------------------------------------------------------------------

def listify(f):
    """Decorator to turn generator results into lists"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return list(f(*args, **kwargs))
    return wrapper

#------------------------------------------------------------------------
# Argument parsing
#------------------------------------------------------------------------

def flatargs(f, args, kwargs, argspec=None):
    """
    Return a single args tuple matching the actual function signature, with
    extraneous args appended to a new tuple 'args' and extraneous keyword
    arguments inserted in a new dict 'kwargs'.

        >>> def f(a, b=2, c=None): pass
        >>> flatargs(f, (1,), {'c':3})
        (1, 2, 3)
        >>> flatargs(f, (), {'a': 1})
        (1, 2, None)
        >>> flatargs(f, (1, 2, 3), {})
        (1, 2, 3)
        >>> flatargs(f, (2,), {'a': 1})
        Traceback (most recent call last):
            ...
        TypeError: f() got multiple values for keyword argument 'a'
    """
    argspec = inspect.getargspec(f) if argspec is None else argspec
    defaults = argspec.defaults or ()
    kwargs = dict(kwargs)

    def unreachable():
        f(*args, **kwargs)
        assert False, "unreachable"

    if argspec.varargs or argspec.keywords:
        raise TypeError("Variable arguments or keywords not supported")

    # -------------------------------------------------
    # Validate argcount

    if (len(args) < len(argspec.args) - len(defaults) - len(kwargs) or
            len(args) > len(argspec.args)):
        # invalid number of arguments
        unreachable()

    # -------------------------------------------------

    # Insert defaults

    tail = min(len(defaults), len(argspec.args) - len(args))
    if tail:
        for argname, default in zip(argspec.args[-tail:], defaults[-tail:]):
            kwargs.setdefault(argname, default)

    # Parse defaults
    extra_args = []
    for argpos in range(len(args), len(argspec.args)):
        argname = argspec.args[argpos]
        if argname not in kwargs:
            unreachable()

        extra_args.append(kwargs[argname])
        kwargs.pop(argname)

    # -------------------------------------------------

    if kwargs:
        unreachable()

    return args + tuple(extra_args)


def alpha_equivalent(spec1, spec2):
    """
    Return whether the inspect argspec `spec1` and `spec2` are equivalent
    modulo naming.
    """
    return (len(spec1.args) == len(spec2.args) and
            bool(spec1.varargs) == bool(spec2.varargs) and
            bool(spec1.keywords) == bool(spec2.keywords))

# ______________________________________________________________________

if __name__ == '__main__':
    import doctest
    doctest.testmod()