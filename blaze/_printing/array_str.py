from __future__ import absolute_import

from . import _arrayprint

def array_str(a):
    return _arrayprint.array2string(a._data)
