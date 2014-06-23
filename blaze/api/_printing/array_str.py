from __future__ import absolute_import, division, print_function

from . import _arrayprint


def array_str(a):
    return _arrayprint.array2string(a.ddesc)
