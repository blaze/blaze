# -*- coding: utf-8 -*-

"""
Python evaluation of blaze AIR.
"""

from __future__ import print_function, division, absolute_import

from blaze.py2help import dict_iteritems

from pykit import types
from pykit.ir import Function, Builder, Value, Op, interp

# Use numpy for now until dynd supports reshape
import numpy as np


def op_kernel(op, *args):
    kernel    = op.metadata['kernel']
    func      = op.metadata['func']
    signature = op.metadata['signature']
    return func(*args)

def op_convert(op, arg):
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

handlers = {
    'kernel': op_kernel,
    'convert': op_convert,
}

def py_interp(func, env=None, exc_model=None, args=()):
    args = [np.array(arg) for arg in args]
    env = env or {}
    env.setdefault('interp.handlers', {}).update(handlers)
    return interp.run(func, env, exc_model, args=args)

run = py_interp