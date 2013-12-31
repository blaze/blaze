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
    jit_env = dict(root_jit_env)
    jitter(func, jit_env)
    treebuilder(func, jit_env)
    ckernel_transformer(func, jit_env)
    return func, env


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
    v = KernelJitter(func)
    visit(v, func)
    v = ConvertJitter(func, v.jitted)
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

    # Delete dead ops in reverse dominating order, so as to only delete ops
    # with 0 live uses
    for op in reversed(transformer.delete_later):
        op.delete()


#------------------------------------------------------------------------
# Jit kernels
#------------------------------------------------------------------------
class KernelJitter(object):
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


class ConvertJitter(object):
    """
    Jit convert ops. Produces a dict `jitted` that maps Operations to jitted
    BlazeElementKernels
    """

    def __init__(self, func, jitted):
        self.func = func
        self.jitted = jitted

    def op_convert(self, op):
        # If all the uses of this convert op have been jitted,
        # then also jit this op
        if all(use in self.jitted for use in self.func.uses[op]):
            dtype = op.type.measure
            blaze_func = make_blazefunc(converter(dtype, op.args[0].type))
            self.jitted[op] = blaze_func


def construct_blaze_kernel(function, overload):
    """
    Parameters
    ==========
    function: blaze.function.BlazeFunc
    overload: blaze.overloading.Overload
    """
    monosig = overload.resolved_sig
    argtypes = monosig.argtypes

    # Try a numba implementation
    py_func, signature = find_impl(function, 'numba', argtypes, monosig)
    if py_func is not None:
        nb_argtypes = [to_numba(a.measure) for a in signature.argtypes]
        nb_restype = to_numba(signature.restype.measure)
        return frompyfunc(py_func, (nb_argtypes, nb_restype), argtypes)

    # Try an LLVM implementation
    py_func, signature = find_impl(function, 'numba', argtypes, monosig)
    if py_func is not None:
        return BlazeElementKernel(py_func, signature.argtypes)


# TODO: factor this out into a "resolve_kernels" or somesuch pass
def find_impl(function, impl_kind, argtypes, expected_signature):
    if function.matches(impl_kind, argtypes):
        overload = function.best_match(impl_kind, argtypes)
        got_signature = overload.resolved_sig

        # Assert agreeable types for now
        # TODO: insert conversions if implementation disagrees

        assert got_signature == expected_signature, (got_signature,
                                                     expected_signature)

        return overload.func, got_signature

    return None, None


#------------------------------------------------------------------------
# Fuse jitted kernels
#------------------------------------------------------------------------
def tree_arg(type):
    kind = llvm_array.SCALAR
    rank = 0
    llvmtype = to_numba(type.measure).to_llvm()
    tree = Argument(type, kind, rank, llvmtype)
    return tree


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
        for arg in func.args:
            tree = tree_arg(arg.type)
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
        jitted = self.jitted.get(op)
        if not jitted:
            return op

        consumers = self.func.uses[op]
        if len(consumers) > 1:
            # We have multiple consumers
            pass

        # TODO: Check external references in metadata in order to determine
        #       whether this is a fusion boundary

        def add_arg(arg, tree):
            self.arguments[op].append((arg, tree))
            self.trees[arg] = tree

        children = []
        for i, arg in enumerate(op.args[1:]):
            rank = len(arg.type.shape)
            if arg in self.trees:
                tree = self.trees[arg]
                self.arguments[op].extend(self.arguments[arg])
            elif arg in self.jitted:
                kernel = self.jitted[arg]
                tree = Argument(arg.type, kernel.kinds[i], rank, kernel.argtypes[i])
                add_arg(arg, tree)
            else:
                # Function argument, construct Argument and `kind` (see
                # BlazeElementKernel.kinds)
                if not all(c.metadata['elementwise']
                               for c in consumers if c.opcode == 'kernel'):
                    raise NotImplementedError(
                        "We have non-elementwise consumers that we don't know "
                        "how to deal with")
                tree = tree_arg(arg.type)
                add_arg(arg, tree)

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
        self.delete_later = [] # Ops to delete afterwards

    def op_convert(self, op):
        if op.args[0] in self.trees and op in self.jitted:
            uses = self.func.uses[op]
            if all(u in self.jitted for u in uses):
                self.delete_later.append(op)

    def op_kernel(self, op):
        if op not in self.trees:
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
