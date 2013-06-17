import sys
import itertools

PY3 = sys.version_info[:2] >= (3,0)

if PY3:
    def dict_iteritems(d):
        return d.items().__iter__()
    xrange = range
    izip = zip
    _inttypes = (int,)
    _strtypes = (str,)
    unicode = str
    imap = map
    basestring = str
    import urllib.parse as urlparse
    from collections import Counter
    from functools import reduce
else:
    import __builtin__
    def dict_iteritems(d):
        return d.iteritems()
    xrange = __builtin__.xrange
    from itertools import izip
    unicode = __builtin__.unicode
    basestring = __builtin__.basestring
    reduce = __builtin__.reduce

    _strtypes = (str, unicode)

    _inttypes = (int, long)
    imap = itertools.imap
    import urlparse
    if sys.version_info >= (2, 7):
        from collections import Counter
    else:
        from .counter_py26 import Counter

if sys.version_info[:2] >= (2, 7):
    from ctypes import c_ssize_t
    from unittest import skip, skipIf
else:
    import ctypes
    if ctypes.sizeof(ctypes.c_void_p) == 4:
        c_ssize_t = ctypes.c_int32
    else:
        c_ssize_t = ctypes.c_int64
    from nose.plugins.skip import SkipTest
    class skip(object):
        def __init__(self, reason):
            self.reason = reason

        def __call__(self, func):
            from nose.plugins.skip import SkipTest
            def wrapped(*args, **kwargs):
                raise SkipTest("Test %s is skipped because: %s" %
                                (func.__name__, self.reason))
            wrapped.__name__ = func.__name__
            return wrapped
    class skipIf(object):
        def __init__(self, condition, reason):
            self.condition = condition
            self.reason = reason

        def __call__(self, func):
            if self.condition:
                from nose.plugins.skip import SkipTest
                def wrapped(*args, **kwargs):
                    raise SkipTest("Test %s is skipped because: %s" %
                                    (func.__name__, self.reason))
                wrapped.__name__ = func.__name__
                return wrapped
            else:
                return func

