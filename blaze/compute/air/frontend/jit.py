# -*- coding: utf-8 -*-

"""
Use blaze.bkernel to assemble ckernels for evaluation.
"""

from __future__ import absolute_import, division, print_function

import collections

from pykit.ir import visit, transform, Op
from datashape.util import to_numba
from numba import jit

from ...bkernel.blaze_kernels import frompyfunc, BlazeElementKernel
from ...bkernel.kernel_tree import Argument, KernelTree
from ... import llvm_array


#------------------------------------------------------------------------
# Interpreter
#------------------------------------------------------------------------


def run(func, env):
    if True or env['strategy'] != 'jit':
        return

    # Identify the nodes to JIT
    jitted = identify_jitnodes(func, env)
    # Build up trees for all those nodes
    trees, arguments = build_kerneltrees(func, jitted)
    # JIT compile the nodes with maximal trees, deleting
    # all their children
    build_ckernels(func, jitted, trees, arguments)
    return func, env

#------------------------------------------------------------------------
# Environment
#------------------------------------------------------------------------

root_jit_env = {
    'jit.jitted':       None, # Jitted kernels, { Op : BlazeElementKernel }
    'jit.trees':        None, # (partial) kernel tree, { Op : KernelTree }
    'jit.arguments':    None, # Accumulated arguments, { Op : [ Op ] }
}


#------------------------------------------------------------------------
# Pipeline
#------------------------------------------------------------------------

def identify_jitnodes(func, env):
    """
    Identifies which nodes (kernels and converters) should be jitted,
    and creates a dictionary mapping them to corresponding
    BlazeElementKernels.
    """
    v = IdentifyJitKernels(func, env)
    visit(v, func)
    v = IdentifyJitConvertors(func, v.jitted)
    visit(v, func)
    return v.jitted


def build_kerneltrees(func, jitted):
    """
    Builds the kernel trees for all the nodes that are to be jitted,
    also populating lists of arguments.
    """
    fuser = BuildKernelTrees(func, jitted)
    visit(fuser, func)
    return fuser.trees, fuser.arguments


def build_ckernels(func, jitted, trees, arguments):
    transformer = BuildCKernels(func, jitted, trees, arguments)
    transform(transformer, func)

    # Delete dead ops in reverse dominating order, so as to only delete ops
    # with 0 live uses
    for op in reversed(transformer.delete_later):
        op.delete()


#------------------------------------------------------------------------
# Jit kernels
#------------------------------------------------------------------------

class IdentifyJitKernels(object):
    """
    Determine which kernels may be jitted. Produces a dict `self.jitted`
    that maps Operations to BlazeElementKernels implementations.
    """

    def __init__(self, func, env):
        self.func = func
        self.env = env

        self.strategies = self.env['strategies']
        self.overloads = self.env['kernel.overloads']

        self.jitted = {}

    def op_kernel(self, op):
        strategy = self.strategies[op]
        if strategy != 'jit':
            return

        overload = self.overloads.get((op, strategy))
        if overload is not None:
            py_func, signature = overload
            blaze_element_kernel = construct_blaze_kernel(py_func, signature)
            self.jitted[op] = blaze_element_kernel


def construct_blaze_kernel(py_func, signature):
    """
    Parameters
    ==========
    function: blaze.function.BlazeFunc
    overload: blaze.overloading.Overload
    """
    nb_argtypes = [to_numba(a.measure) for a in signature.argtypes]
    nb_restype = to_numba(signature.restype.measure)
    return frompyfunc(py_func, (nb_argtypes, nb_restype), signature.argtypes)


class IdentifyJitConvertors(object):
    """
    Determine which conversion operators should be jitted. Adds to the dict
    `self.jitted` that maps Operations to BlazeElementKernels implementations
    """

    def __init__(self, func, jitted):
        self.func = func
        self.jitted = jitted

    def op_convert(self, op):
        # If all the uses of this convert op have been jitted,
        # then also jit this op
        if all(use in self.jitted for use in self.func.uses[op]):
            dtype = op.type.measure
            blaze_func = BlazeElementKernel(converter(dtype, op.args[0].type).lfunc)
            self.jitted[op] = blaze_func


#------------------------------------------------------------------------
# Fuse jitted kernels
#------------------------------------------------------------------------

def leaf_arg(type):
    kind = llvm_array.SCALAR
    rank = 0
    llvmtype = to_numba(type.measure).to_llvm()
    tree = Argument(type, kind, rank, llvmtype)
    return tree


class BuildKernelTrees(object):
    """
    Build KernelTrees from the BlazeElementKernels in 'jitted'.
    """

    def __init__(self, func, jitted):
        self.func = func
        self.jitted = jitted
        self.trees = {}
        self.arguments = collections.defaultdict(list)

        # Start off with the function arguments as leaves for the trees
        for arg in func.args:
            tree = leaf_arg(arg.type)
            self.trees[arg] = tree
            self.arguments[arg] = [(arg, tree)]

    def op_convert(self, op):
        # TODO: Rewrite 'convert' ops before any of this stuff to kernel appl
        arg = op.args[0]
        if op in self.jitted and arg in self.trees:
            children = [self.trees[arg]]
            self.trees[op] = KernelTree(self.jitted[op], children)
            self.arguments[op] = list(self.arguments[arg])

    def op_kernel(self, op):
        elementkernel = self.jitted.get(op)
        if not elementkernel:
            return op

        consumers = self.func.uses[op]
        if len(consumers) > 1:
            # This Op has multiple consumers
            pass

        # TODO: Check external references in metadata in order to determine
        #       whether this is a fusion boundary

        children = []
        for i, arg in enumerate(op.args[1:]):
            rank = len(arg.type.shape)
            if arg in self.trees:
                # This argument already has a tree, include it
                # as a subtree
                tree = self.trees[arg]
                self.arguments[op].extend(self.arguments[arg])
            elif arg in self.jitted:
                # TODO: the original code here doesn't work, shuffling things
                #       around introduce unexpected LLVM bitcast exceptions
                raise RuntimeError('internal error in blaze JIT code with arg %r', arg)
            else:
                # This argument is a non-jittable kernel, add it as a leaf node
                if not all(c.metadata['elementwise']
                               for c in consumers if c.opcode == 'kernel'):
                    raise NotImplementedError(
                        "We have non-elementwise consumers that we don't know "
                        "how to deal with")
                tree = leaf_arg(arg.type)
                self.trees[arg] = tree
                self.arguments[arg] = [(arg, tree)]
                self.arguments[op].append((arg, tree))

            children.append(tree)

        self.trees[op] = KernelTree(elementkernel, children)

#------------------------------------------------------------------------
# Rewrite to CKernels
#------------------------------------------------------------------------

class BuildCKernels(object):

    def __init__(self, func, jitted, trees, arguments):
        self.func = func
        self.jitted = jitted
        self.trees = trees
        self.arguments = arguments
        self.delete_later = [] # Ops to delete afterwards

    def op_convert(self, op):
        if op.args[0] in self.trees and op in self.jitted:
            uses = self.func.uses[op]
            if all(u in self.jitted for u in uses):
                self.delete_later.append(op)

    def op_kernel(self, op):
        if op not in self.jitted:
            return op

        uses = self.func.uses[op]

        if all(u in self.trees for u in uses):
            # All our consumers know about us and have us as an argument
            # in their tree! Delete this op, only the root will perform a
            # rewrite.
            self.delete_later.append(op)
        elif any(u in self.trees for u in uses):
            # Some consumers have us as a node, but others don't. This
            # forms a ckernel boundary, so we need to detach ourselves!
            raise NotImplementedError
        else:
            # No consumer has us as an internal node in the kernel tree, we
            # are a kerneltree root
            tree = self.trees[op]
            # out_rank = len(op.type.shape)
            # tree = tree.adapt(out_rank, llvm_array.C_CONTIGUOUS)
            ckernel_deferred = tree.make_ckernel_deferred(op.type)

            # Flatten the tree of args, removing duplicates just
            # as the kernel tree does.
            args = []
            for arg in op.args[1:]: # Skip op.args[0], the kernel string name
                for ir_arg, kt_arg in self.arguments[arg]:
                    if ir_arg not in args:
                        args.append(ir_arg)

            new_op = Op('ckernel', op.type, [ckernel_deferred, args], op.result)
            new_op.add_metadata({'rank': 0,
                                 'parallel': True})
            return new_op


#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def make_blazefunc(f):
    #return BlazeFuncDeprecated(f.__name__, template=f)
    return BlazeElementKernel(f.lfunc)


def converter(blaze_dtype, blaze_argtype):
    """
    Generate an element-wise conversion function that numba can jit-compile.
    """
    dtype = to_numba(blaze_dtype.measure)
    argtype = to_numba(blaze_argtype.measure)

    @jit(dtype(argtype))
    def convert(value):
        return dtype(value)

    return convert


def make_ckernel(blaze_func):
    raise NotImplementedError
