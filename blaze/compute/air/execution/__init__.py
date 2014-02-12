# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from blaze.compute.strategy import PY, JIT
from . import python_interp
from . import jit_interp

interpreters = {
    PY:  python_interp,
    JIT: jit_interp,
}
lookup_interp = interpreters.__getitem__

def register_interp(interp_kind, interpreter):
    interpreters[interp_kind] = interpreter