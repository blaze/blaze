# -*- coding: utf-8 -*-

"""
Python evaluation of blaze AIR.
"""

from __future__ import absolute_import, division, print_function

from pykit.ir import interp

import blaze

# Use numpy for now until dynd supports reshape
import numpy as np

#------------------------------------------------------------------------
# Interpreter
#------------------------------------------------------------------------

def interpret(func, env, args, **kwds):
    args = [np.array(arg) for arg in args]
    env = {'interp.handlers' : handlers}
    result = interp.run(func, env, None, args=args)
    return blaze.array(result)

#------------------------------------------------------------------------
# Handlers
#------------------------------------------------------------------------

def op_pykernel(interp, func, args):
    return func(*args)

def op_convert(interp, arg):
    op = interp.op
    dshape = op.type

    # Broadcasting
    out_shape = arg.shape
    in_shape = dshape.shape

    for i in range(len(out_shape), len(in_shape)):
        out_shape = (1,) + out_shape

    # Reshape with the output shape, since it may have broadcasting dimensions
    arg = arg.reshape(out_shape)

    # Dtype conversion
    in_dtype = dshape.measure.to_numpy_dtype()
    if arg.dtype != in_dtype:
        arg = arg.astype(in_dtype)

    return arg

def op_ret(interp, arg):
    interp.pc = -1
    return arg

handlers = {
    'pykernel':   op_pykernel,
    'convert':  op_convert,
    'ret':      op_ret,
}
