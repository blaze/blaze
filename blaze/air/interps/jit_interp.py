# -*- coding: utf-8 -*-

"""
JIT evaluation of blaze AIR.
"""

from __future__ import print_function, division, absolute_import

from ..pipeline import run_pipeline
from ..passes import jit, ckernel, allocation
from .ckernel_interp import (interpret as ckernel_interpret,
                             compile as ckernel_compile)

#------------------------------------------------------------------------
# Interpreter
#------------------------------------------------------------------------

def compile(func, env):
    func, env = run_pipeline(func, env, passes)
    return ckernel_compile(func, env)

interpret = ckernel_interpret

#------------------------------------------------------------------------
# Passes
#------------------------------------------------------------------------

passes = [
    jit,
]