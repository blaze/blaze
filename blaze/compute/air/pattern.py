"""
Taken and slightly adapted from lair/backend/pattern.py
"""

from __future__ import print_function, division, absolute_import

import inspect
from functools import partial
from collections import namedtuple

Case = namedtuple('Case', ['pattern', 'function', 'argspec'])

class match(object):
    '''Dispatch a function by pattern matching.
    Start with the most generic case; Add more specialized cases afterwards.
    Pattern value of type `type` is matched using the builtin `isinstance`.
    Pattern value of type `Matcher` is used directly.
    Pattern value of other types is matched using `==`.
    '''
    def __init__(self, func):
        self._generic = func
        self._cases = []
        self._argspec = inspect.getargspec(func)

        assert not self._argspec.varargs, 'Thou shall not use *args'
        assert not self._argspec.keywords, 'Thou shall not use **kws'

    def case(self, *patargs, **patkws):
        def wrap(fn):
            argspec = inspect.getargspec(fn)
            assert len(argspec.args) == len(self._argspec.args), "mismatch signature"
            for pat, arg in zip(patargs, argspec.args):
                assert arg not in patkws
                patkws[arg] = pat
            case = Case(_prepare_pattern(patkws.items()), fn, argspec)
            self._cases.append(case)
            return self
        return wrap

    def __get__(self, inst, type=None):
        return partial(self, inst)

    def __call__(self, *args, **kwds):
        for case in reversed(self._cases):
            kws = dict(kwds)
            _pack_args(case.argspec, args, kws)
            for k, matcher in case.pattern:
                if not matcher(kws[k]):
                    break
            else:
                return case.function(*args, **kwds)
        else:
            return self._generic(*args, **kwds)


def _pack_args(argspec, args, kws):
    args = list(args)
    for v, k in zip(args, argspec.args):
        if k in kws:
            return NameError(k)
        else:
            kws[k] = v


def _prepare_pattern(pats):
    return tuple((k, _select_matcher(v)) for k, v in pats)


def _select_matcher(v):
    if isinstance(v, Matcher):
        return v
    elif isinstance(v, type):
        return InstanceOf(v)
    else:
        return Equal(v)


class Matcher(object):
    __slots__ = 'arg'
    def __init__(self, arg):
        self.arg = arg


class InstanceOf(Matcher):
    def __call__(self, x):
        return isinstance(x, self.arg)


class Equal(Matcher):
    def __call__(self, x):
        return self.arg == x


class Custom(Matcher):
    def __call__(self, x):
        return self.arg(x)

custom = Custom     # alias
