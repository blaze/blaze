# -*- coding: utf-8 -*-

"""
Python evaluation of blaze AIR.
"""

from __future__ import print_function, division, absolute_import
from pykit.ir import interp

import blaze

# Use numpy for now until dynd supports reshape
import numpy as np

#------------------------------------------------------------------------
# Interpreter
#------------------------------------------------------------------------

def py_interp(func, args, **kwds):
    args = [np.array(arg) for arg in args]
    env = {'interp.handlers' : handlers}
    result = interp.run(func, env, None, args=args)
    return blaze.array(result)

def compile(func, env):
    return func

run = py_interp

#------------------------------------------------------------------------
# Handlers
#------------------------------------------------------------------------

def op_kernel(interp, funcname, *args):
    op = interp.op

    kernel   = op.metadata['kernel']
    overload = op.metadata['overload']

    func = overload.func
    return func(*args)

def op_convert(interp, arg):
    op = interp.op
    dshape = op.type

    # Broadcasting
    out_shape = arg.shape
    in_shape = dshape.shape
    if len(out_shape) < len(in_shape):
        arg = arg.reshape(in_shape)

    # Dtype conversion
    in_dtype = dshape.measure.to_numpy_dtype()
    if arg.dtype != in_dtype:
        arg = arg.astype(in_dtype)

    return arg

def op_ret(interp, arg):
    interp.pc = -1
    return arg

handlers = {
    'kernel':   op_kernel,
    'convert':  op_convert,
    'ret':      op_ret,
}
