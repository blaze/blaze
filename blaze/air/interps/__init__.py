# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from . import python_interp
from . import jit_interp

interpreters = {
    'py': python_interp,
    'jit': jit_interp,
}
lookup_interp = interpreters.__getitem__

def register_interp(interp_kind, interpreter):
    interpreters[interp_kind] = interpreter