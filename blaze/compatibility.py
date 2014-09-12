from __future__ import absolute_import, division, print_function

import sys
PY3 = sys.version_info[0] == 3

if PY3:
    from urllib.request import urlopen
    import builtins as builtins
    def apply(f, args):
        return f(*args)

else:
    from urllib2 import urlopen
    import __builtin__ as builtins
    apply = apply

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
from toolz.compatibility import map, zip, range, reduce

if PY2:
    _strtypes = (str, unicode)
    _inttypes = (int, long)
    unicode = unicode
    basestring = basestring
else:
    _inttypes = (int,)
    _strtypes = (str,)
    unicode = str
    basestring = str


import io


try:
    SEEK_END = io.SEEK_END
except AttributeError:
    SEEK_END = 2


try:
    import pytest
    skipif = pytest.mark.skipif
    xfail = pytest.mark.xfail
    min_python_version = skipif(sys.version_info < (2, 7),
                                reason="Python >= 2.7 required")
    raises = pytest.raises
except ImportError:
    # TODO: move the above into a separate testing utils module
    pass


if sys.version_info >= (2, 7):
    from ctypes import c_ssize_t
else:
    import ctypes
    if ctypes.sizeof(ctypes.c_void_p) == 4:
        c_ssize_t = ctypes.c_int32
    else:
        c_ssize_t = ctypes.c_int64


try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
