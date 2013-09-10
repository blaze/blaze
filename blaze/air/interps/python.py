# -*- coding: utf-8 -*-

"""
Python evaluation of blaze AIR.
"""

from __future__ import print_function, division, absolute_import
from pykit.ir import interp

# Use numpy for now until dynd supports reshape
import numpy as np

#------------------------------------------------------------------------
# Interpreter
#------------------------------------------------------------------------

def py_interp(func, env=None, exc_model=None, args=()):
    args = [np.array(arg) for arg in args]
    env = env or {}
    env.setdefault('interp.handlers', {}).update(handlers)
    return interp.run(func, env, exc_model, args=args)

run = py_interp

#------------------------------------------------------------------------
# Handlers
#------------------------------------------------------------------------

def op_kernel(interp, funcname, *args):
    op = interp.op

    kernel    = op.metadata['kernel']
    func      = op.metadata['func']
    signature = op.metadata['signature']

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
