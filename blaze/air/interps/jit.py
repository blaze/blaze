# -*- coding: utf-8 -*-

"""
Use blaze.bkernel to assemble ckernels for evaluation.
"""

from __future__ import print_function, division, absolute_import
import collections

from pykit.ir import interp, visit, transform, Op

import blaze
from blaze.bkernel import BlazeFuncDeprecated
from blaze.bkernel.blaze_kernels import frompyfunc, fromctypes, BlazeElementKernel
from blaze.bkernel.kernel_tree import Argument, KernelTree
from blaze.datashape.util import to_numba

from blaze import llvm_array

#------------------------------------------------------------------------
# Interpreter
#------------------------------------------------------------------------

def compile(func, env):
    # NOTE: A problem of using a DataDescriptor as part of KernelTree is that
    #       we can now only compile kernels when we have actual data. This is
    #       a problem for offline compilation strategies.
    jit_env = dict(root_jit_env)
    jitter(func, jit_env)
    treebuilder(func, jit_env)
    ckernel_transformer(func, jit_env)
    return func

def run(func, args, **kwds):
    deferred_array = jit_interp(func, args=args)
    result = blaze.eval(deferred_array, **kwds)
    return result

#------------------------------------------------------------------------
# Environment
#------------------------------------------------------------------------

root_jit_env = {
    'jitted':       None, # Jitted kernels, { Op : BlazeElementKernel }
    'trees':        None, # (partial) kernel tree, { Op : KernelTree }
    'arguments':    None, # Accumulated arguments, { Op : [ Op ] }
}

#------------------------------------------------------------------------
# Pipeline
#------------------------------------------------------------------------

def jitter(func, jit_env):
    v = Jitter(func)
    visit(v, func)
    jit_env['jitted'] = v.jitted

def treebuilder(func, jit_env):
    fuser = JitFuser(func, jit_env['jitted'])
    visit(fuser, func)
    jit_env['trees'] = fuser.trees
    jit_env['arguments'] = fuser.arguments

def ckernel_transformer(func, jit_env):
    transformer = CKernelTransformer(func, jit_env['jitted'],
                                     jit_env['trees'], jit_env['arguments'])
    transform(transformer, func)
    for op in transformer.delete:
        op.delete()

#------------------------------------------------------------------------
# Jit kernels
#------------------------------------------------------------------------

class Jitter(object):
    """
    Jit kernels. Produces a dict `jitted` that maps Operations to jitted
    BlazeElementKernels
    """

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

#------------------------------------------------------------------------
# Fuse jitted kernels
#------------------------------------------------------------------------

class JitFuser(object):
    """
    Build KernelTrees from jitted BlazeElementKernels.
    """

    def __init__(self, func, jitted):
        self.func = func
        self.jitted = jitted
        self.trees = {}

        # Accumulated arguments (registers) for a kerneltree
        self.arguments = collections.defaultdict(list)

    def op_kernel(self, op):
        jitted = self.jitted.get(op)
        if not jitted:
            return op

        consumers = self.func.uses[op]
        if len(consumers) > 1:
            # We have multiple consumers
            pass

        # TODO: Check external references in metadata in order to determine
        #       whether this is a fusion boundary

        children = []
        for i, arg in enumerate(op.args[1:]):
            rank = len(arg.type.shape)
            if arg in self.trees:
                tree = self.trees[arg]
                self.arguments[op].extend(self.arguments[arg])
            elif arg in self.jitted:
                kernel = self.jitted[arg]
                tree = Argument(arg.type, kernel.kinds[i], rank, kernel.argtypes[i])
                self.arguments[op].append((arg, tree))
            else:
                # Function argument, construct Argument and `kind` (see
                # BlazeElementKernel.kinds)
                if not all(c.metadata['elementwise']
                               for c in consumers if c.opcode == 'kernel'):
                    raise NotImplementedError(
                        "We have non-elementwise consumers that we don't know "
                        "how to deal with")
                kind = llvm_array.SCALAR
                rank = 0
                llvmtype = to_numba(arg.type.measure).to_llvm()
                tree = Argument(arg.type, kind, rank, llvmtype)
                self.arguments[op].append((arg, tree))

            children.append(tree)

        self.trees[op] = KernelTree(jitted, children)

#------------------------------------------------------------------------
# Rewrite to CKernels
#------------------------------------------------------------------------

class CKernelTransformer(object):

    def __init__(self, func, jitted, trees, arguments):
        self.func = func
        self.jitted = jitted
        self.trees = trees
        self.arguments = arguments
        self.delete = set() # Ops to delete afterwards

    def op_kernel(self, op):
        if op not in self.trees:
            return op

        uses = self.func.uses[op]

        if all(u in self.trees for u in uses):
            # All our consumers know about us and have us as an argument
            # in their tree! Delete this op, only the root will perform a
            # rewrite.
            self.delete.add(op)
        elif any(u in self.trees for u in uses):
            # Some consumers have us as a node, but others don't. This
            # forms a ckernel boundary, so we need to detach ourselves!
            raise NotImplementedError
        else:
            # No consumer has us as an internal node in the kernel tree, we
            # are a kerneltree root
            tree = self.trees[op]
            unbound_ckernel = tree.make_unbound_ckernel(strided=False)
            # Skip kernel string name, first arg to 'kernel' Operations
            args = [ir_arg for arg in op.args[1:]
                               for ir_arg, kt_arg in self.arguments[arg]]
            return Op('ckernel', op.type, [unbound_ckernel, args], op.result)


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
