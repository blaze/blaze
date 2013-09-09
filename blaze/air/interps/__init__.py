# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from . import python
from . import jit

interpreters = {
    'py': python,
    'jit': jit,
}
lookup_interp = interpreters.__getitem__