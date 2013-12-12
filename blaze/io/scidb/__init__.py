# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from .constructors import empty, zeros, ones, handle
from .kernel import scidb_elementwise, scidb_function, scidb_kernel, AFL, AQL
from . conn import connect
from . import scidb_interp

# --- Initialize ---
from . import ufuncs

# Register scidb AIR interpreter
from blaze.compute.air import interps
interps.register_interp('scidb', scidb_interp)