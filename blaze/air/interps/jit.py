# -*- coding: utf-8 -*-

"""
Use blaze.bkernel to assemble ckernels for evaluation.
"""

from __future__ import print_function, division, absolute_import
from pykit.ir import interp

import blaze
from blaze.bkernel import BlazeFunc
from blaze.datashape.util import to_numba

#------------------------------------------------------------------------
# Interpreter
#------------------------------------------------------------------------

def jit_interp(func, env=None, exc_model=None, args=()):
    env = env or {}
    env.setdefault('interp.handlers', {}).update(handlers)
    return interp.run(func, env, exc_model, args=args)


def compile(func, env):
    # NOTE: A problem of using a DataDescriptor as part of KernelTree is that
    #       we can now only compile kernels when we have actual data. This is
    #       a problem for offline compilation strategies.
    return func

def run(func, args):
    deferred_array = jit_interp(func, args=args)
    result = blaze.eval(deferred_array)
    return result

#------------------------------------------------------------------------
# Handlers
#------------------------------------------------------------------------

def op_kernel(interp, funcname, *args):
    op = interp.op
    overload = op.metadata['overload']
    func = overload.func

    blaze_func = make_blazefunc(func)
    return blaze_func(*args)

def op_convert(interp, arg):
    op = interp.op
    dtype = op.type.measure
    blaze_func = make_blazefunc(converter(dtype))
    return blaze_func(arg)

def op_ret(interp, arg):
    interp.pc = -1
    return arg

handlers = {
    'kernel':   op_kernel,
    'convert':  op_convert,
    'ret':      op_ret,
}

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def make_blazefunc(f):
    return BlazeFunc(f.__name__, template=f)

def converter(blaze_type):
    """
    Generate an element-wise conversion function that numba can jit-compile.
    """
    T = to_numba(blaze_type)
    def convert(value):
        return T(value)
    return convert