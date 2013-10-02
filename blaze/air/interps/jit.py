# -*- coding: utf-8 -*-

"""
Use blaze.bkernel to assemble ckernels for evaluation.
"""

from __future__ import print_function, division, absolute_import
from pykit.ir import interp, visit

import blaze
from blaze.bkernel import BlazeFuncDeprecated
from blaze.bkernel.blaze_kernels import frompyfunc, fromctypes, BlazeElementKernel
from blaze.bkernel.kernel_tree import Argument, KernelTree
from blaze.datashape.util import to_numba

#------------------------------------------------------------------------
# Interpreter
#------------------------------------------------------------------------

def compile(func, env):
    # NOTE: A problem of using a DataDescriptor as part of KernelTree is that
    #       we can now only compile kernels when we have actual data. This is
    #       a problem for offline compilation strategies.
    jitted = jitter(func, env)
    condense_jitted_kernels(func, env, jitted)
    return func

def run(func, args, **kwds):
    deferred_array = jit_interp(func, args=args)
    result = blaze.eval(deferred_array, **kwds)
    return result

#------------------------------------------------------------------------
# Jit kernels
#------------------------------------------------------------------------

class Jitter(object):
    """Jit kernels"""

    def __init__(self, func):
        self.func = func
        self.jitted = {}

    def op_kernel(self, op):
        function = op.metadata['kernel']
        overload = op.metadata['overload']
        impl = construct_blaze_kernel(function, overload)
        if impl is not None:
            self.jitted[op] = impl

    def op_convert(self, op):
        # TODO: Use a heuristic to see whether we need to handle this, or
        #       someone else does
        [arg] = op.args

        dtype = op.type.measure
        blaze_func = make_blazefunc(converter(dtype))
        self.jitted[op] = blaze_func(arg)


def construct_blaze_kernel(function, overload):
    """
    Parameters
    ==========
    function: blaze.kernel.Kernel
    overload: blaze.overloading.Overload
    """
    func = overload.func
    polysig = overload.sig
    monosig = overload.resolved_sig

    numba_impls = function.find_impls(func, polysig, 'numba')
    llvm_impls = function.find_impls(func, polysig, 'llvm')

    if numba_impls:
        [impl] = numba_impls
        argtypes = [to_numba(a.measure) for a in monosig.argtypes]
        restype = to_numba(monosig.restype.measure)
        return frompyfunc(impl, (argtypes, restype), monosig.argtypes)
    elif llvm_impls:
        [impl] = llvm_impls
        return BlazeElementKernel(impl, monosig.argtypes)
    else:
        return None


def jitter(func, env=None):
    v = Jitter(func)
    visit(v, func)
    return v.jitted

#------------------------------------------------------------------------
# Fuse jitted kernels
#------------------------------------------------------------------------

class JitFuser(object):

    def __init__(self, func, jitted):
        self.func = func
        self.jitted = jitted
        self.trees = {}

    def op_kernel(self, op):
        function = op.metadata['kernel']
        overload = op.metadata['overload']
        jitted = self.jitted.get(op)
        if not jitted:
            return op

        consumers = self.func.uses[op]
        if len(consumers) > 1:
            # We have multiple consumers
            pass

        # TODO: Check external references in metadata in order to determine
        #       whether this is a fusion boundary

        args = []
        for arg in op.args[1:]:
            if arg in self.trees:
                tree = self.trees[arg]
            else:
                kernel = self.jitted[arg]
                tree_arg = Argument(arg, kernel.kinds[i],
                                    self.ranks[i], kernel.argtypes[i])
                children.append(tree_arg)


def condense_jitted_kernels(func, env, jitted):
    visit(JitFuser(func, jitted), func)

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def make_blazefunc(f):
    return BlazeFuncDeprecated(f.__name__, template=f)

def converter(blaze_type):
    """
    Generate an element-wise conversion function that numba can jit-compile.
    """
    T = to_numba(blaze_type)
    def convert(value):
        return T(value)
    return convert

def make_ckernel(blaze_func):
    raise NotImplementedError
