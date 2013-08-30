# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import string
import inspect
import functools
import collections

try:
    from collections import MutableMapping
except ImportError as e:
    # Python 3
    from UserDict import DictMixin as MutableMapping

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

def flatargs(f, args, kwargs):
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
    argspec = inspect.getargspec(f)
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


#------------------------------------------------------------------------
# Data Structures
#------------------------------------------------------------------------

class IdentityDict(MutableMapping):
    """
    Map mapping objects on identity to values

        >>> d = IdentityDict({'a': 2, 'b': 3})
        >>> sorted(d.items())
        [('a', 2), ('b', 3)]

        >>> class AlwaysEqual(object):
        ...     def __eq__(self, other):
        ...         return True
        ...     def __repr__(self):
        ...         return "eq"
        ...
        >>> x, y = AlwaysEqual(), AlwaysEqual()
        >>> d[x] = 4 ; d[y] = 5
        >>> sorted(d.items())
        [('a', 2), ('b', 3), (eq, 4), (eq, 5)]
    """

    def __init__(self, d=None):
        self.data = {}          # id(key) -> value
        self.ks = []            # [key]
        self.update(d or [])

    def __getitem__(self, key):
        try:
            return self.data[id(key)]
        except KeyError:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if id(key) not in self.data:
            self.ks.append(key)
        self.data[id(key)] = value

    def __delitem__(self, key):
        self.ks.remove(key)
        del self.data[id(key)]

    def __repr__(self):
        # This is not correctly implemented in DictMixin for us, since it takes
        # the dict() of iteritems(), merging back equal keys
        return "{ %s }" % ", ".join("%r: %r" % (k, self[k]) for k in self.keys())

    def keys(self):
        return list(self.ks)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        d = cls()
        for key in iterable:
            d[key] = value
        return d

    def __iter__(self):
        return iter(self.ks)

    def __len__(self):
        return len(self.ks)


class IdentitySet(set):
    def __init__(self, it=()):
        self.d = IdentityDict()
        self.update(it)

    def add(self, x):
        self.d[x] = None

    def remove(self, x):
        del self.d[x]

    def update(self, it):
        for x in it:
            self.add(x)

    def __contains__(self, key):
        return key in self.d

#------------------------------------------------------------------------
# Temporary names
#------------------------------------------------------------------------

def make_temper():
    """Return a function that returns temporary names"""
    temps = collections.defaultdict(int)

    def temper(name=""):
        varname = name.rstrip(string.digits)
        count = temps[varname]
        temps[varname] += 1
        if varname and count == 0:
            return varname
        return varname + str(count)

    return temper

def make_stream(seq, _temp=make_temper()):
    """Create a stream of temporaries seeded by seq"""
    while 1:
        for x in seq:
            yield _temp(x)

gensym = make_stream(string.uppercase).next

# ______________________________________________________________________

if __name__ == '__main__':
    import doctest
    doctest.testmod()