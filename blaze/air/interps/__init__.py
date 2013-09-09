# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from . import python
from .python import py_interp

interpreters = {
    'py': python,
}
lookup_interp = interpreters.__getitem__