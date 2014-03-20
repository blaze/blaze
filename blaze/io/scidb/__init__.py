from __future__ import absolute_import, division, print_function

from .constructors import empty, zeros, ones, handle
from .kernel import AFL, AQL, SCIDB
from . conn import connect
from . import scidb_interp

# --- Initialize ---
from . import ufuncs

# Register scidb AIR interpreter
# from blaze.compute.air import execution
# execution.register_interp('scidb', scidb_interp)
