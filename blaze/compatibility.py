from __future__ import absolute_import, division, print_function

import sys
from types import MethodType

import pandas.util.testing as tm

PY3 = sys.version_info[0] == 3
PY2 = sys.version_info[0] == 2

if PY3:
    import builtins

    def apply(f, args, **kwargs):
        return f(*args, **kwargs)
else:
    import __builtin__ as builtins
    apply = builtins.apply


try:
    import cPickle as pickle
except ImportError:
    import pickle


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


from toolz.compatibility import map, zip, range, reduce


if PY2:
    _inttypes = (int, long)
    unicode = builtins.unicode
    basestring = builtins.basestring
    _strtypes = (str, unicode)

    def boundmethod(func, instance):
        return MethodType(func, instance, type(instance))

    from itertools import izip_longest as zip_longest
    from contextlib2 import ExitStack
else:
    _inttypes = (int,)
    _strtypes = (str,)
    unicode = str
    basestring = str
    boundmethod = MethodType

    from itertools import zip_longest
    from contextlib import ExitStack


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


def assert_series_equal(left, right, check_names=True, **kwargs):
    """Backwards compatibility wrapper for
    ``pandas.util.testing.assert_series_equal``

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series(list('abc'), name='a')
    >>> s2 = pd.Series(list('abc'), name='b')
    >>> assert_series_equal(s, s2)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    AssertionError: ...
    >>> assert_series_equal(s, s2, check_names=False)

    See Also
    --------
    pandas.util.testing.assert_series_equal
    """
    try:
        return tm.assert_series_equal(left, right, check_names=check_names,
                                      **kwargs)
    except TypeError:
        if check_names:
            assert left.name == right.name
        return tm.assert_series_equal(left, right, **kwargs)
