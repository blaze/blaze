from __future__ import absolute_import, division, print_function

import sys
PY3 = sys.version_info[0] > 2

if PY3:
    from urllib.request import urlopen
    import builtins as builtins
    def apply(f, args):
        return f(*args)

else:
    from urllib2 import urlopen
    import __builtin__ as builtins
    apply = apply

import sys
import itertools

# Portions of this taken from the six library, licensed as follows.
#
# Copyright (c) 2010-2013 Benjamin Peterson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


PY2 = sys.version_info[0] == 2

if PY2:
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
    map = itertools.imap
    import urlparse
    def exec_(_code_, _globs_=None, _locs_=None):
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec("""exec _code_ in _globs_, _locs_""")
else:
    def dict_iteritems(d):
        return d.items().__iter__()
    xrange = range
    izip = zip
    _inttypes = (int,)
    _strtypes = (str,)
    unicode = str
    map = map
    basestring = str
    import urllib.parse as urlparse
    from functools import reduce
    import builtins
    exec_ = getattr(builtins, "exec")


try:
    import pytest
    skipif = pytest.mark.skipif
    xfail = pytest.mark.xfail
    min_python_version = skipif(sys.version_info < (2, 7),
                                reason="Python >= 2.7 required")
except ImportError:
    pass


if sys.version_info >= (2, 7):
    from ctypes import c_ssize_t
else:
    import ctypes
    if ctypes.sizeof(ctypes.c_void_p) == 4:
        c_ssize_t = ctypes.c_int32
    else:
        c_ssize_t = ctypes.c_int64
