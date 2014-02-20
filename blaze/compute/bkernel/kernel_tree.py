from __future__ import absolute_import, division, print_function

import sys
import string

import llvm.core as lc
from llvm import LLVMException
from .. import llvm_array as lla
from .blaze_kernels import BlazeElementKernel, refresh_name
from . import blaze_kernels
from ... import py2help

def letters(source=string.ascii_lowercase):
    k = 0
    while 1:
        for a in source:
            yield a+str(k) if k else a
        k = k+1


# An Argument to a kernel tree (encapsulates the array, argument kind and rank)
# FIXME --- perhaps we should have _llvmtype be a SCALAR kind or a string that gets converted
#     to the correct llvmtype when needed.
class Argument(object):
    _shape = None
    def __init__(self, dshape, kind, rank, llvmtype):
        self.dshape = dshape
        if isinstance(kind, tuple):
            kind = kind[0]
        self.kind = kind
        self.rank = rank
        self._llvmtype = llvmtype

    #def __eq__(self, other):
    #    if not isinstance(other, Argument):
    #        return NotImplemented
    #    # FIXME: Should remove kind check and cast different kinds
    #    #        in the generated code.
    #    return ((self.dshape is other.dshape) and
    #            (self.rank == other.rank) and
    #            (self.kind==other.kind))

    # FIXME:
    #   Because module linking destroys struct names we need to store
    #   a stringified version of the element type and then
    #   convert as needed...
    @property
    def llvmtype(self):
        self._llvmtype = refresh_name(self._llvmtype)
        return self._llvmtype

    #@property
    #def shape(self):
    #    if self._shape is None:
    #        if self.rank == 0:
    #            self._shape = ()
    #        else:
    #            self._shape = self.arr.dshape.shape[-self.rank:]
    #    return self._shape

    def get_kernel_dshape(self):
        """Returns the kernel data-shape of the argument."""
        rank = self.rank
        total_dshape = self.dshape
        sub = len(total_dshape)-1-rank
        return total_dshape.subarray(sub)

# A KernelTree is just the bare element-wise kernel functions
# (no arguments).  Any arguments are identified as unique-names
# in an abstract name-space
# All nodes in the kernel tree can also be named
#   or else a unique-name
# from the abstract name-space will be created.
# Each KernelTree has a single llvm module name-space
class KernelTree(object):
    _stream_of_unique_names = letters()
    _stream_of_unique_kernels = letters()

    _fused = None
    _mark = False
    _shape = None

    def __init__(self, kernel, children=[], name=None):
        assert isinstance(kernel, BlazeElementKernel)
        for el in children:
            assert isinstance(el, (KernelTree, Argument))
        self.kernel = kernel
        self.children = children
        if name is None:
            name = 'node_' + next(self._stream_of_unique_names)
        self.name = name

    def _reset_marks(self):
        self._mark = False
        if not self.leafnode:
            for child in self.children:
                if isinstance(child, KernelTree):
                    child._reset_marks()

    def sorted_nodes(self):
        """Return depth-first list of unique KernelTree Nodes.

        The root of the tree will be the last node.
        """
        nodes = []
        self._reset_marks()
        self.visit(nodes)
        return nodes

    @property
    def shape(self):
        if self._shape is None:
            shapeargs = [child.shape for child in self.children]
            self._shape = self.kernel.shapefunc(*shapeargs)
        return self._shape

    @property
    def leafnode(self):
        return all(isinstance(child, Argument) for child in self.children)

    def visit(self, nodes):
        if not self.leafnode:
            for child in self.children:
                if isinstance(child, KernelTree):
                    child.visit(nodes)
        if not self._mark:
            nodes.append(self)
            self._mark = True

    def fuse(self):
        if self._fused is not None:
            return self._fused
        # Even if self is a leaf node (self.leafnode is True), do
        # this processing, so as to consistently combine repeated
        # arguments.
        krnlobj, children = fuse_kerneltree(self, self.kernel.module)
        new = KernelTree(krnlobj, children)
        self._update_kernelptrs(new)
        return new

    def _update_kernelptrs(self, eltree):
        self._fused = eltree
        kernel = eltree.kernel

    def make_ckernel_deferred(self, out_dshape):
        return self.fuse().kernel.make_ckernel_deferred(out_dshape)

    def __str__(self):
        pre = self.name + '('
        post = ')'
        strs = []
        for child in self.children:
            if isinstance(child, Argument):
                strs.append('<arg>')
            else:
                strs.append(str(child))
        body = ",".join(strs)
        return pre + body + post

class _cleanup(object):
    def __init__(self, builder, freefunc, freedata):
        self.freefunc = freefunc
        self.freedata = freedata
        self.builder = builder

    def _dealloc(self):
        self.builder.call(self.freefunc, [self.freedata])

# This modifies the node to add a reference the output as llvm_obj
def insert_instructions(node, builder, output=None):
    kernel = node.kernel
    is_scalar = (kernel.kinds[-1] == lla.SCALAR)
    #allocate space for output if necessary
    new = None
    if output is None:
        if kernel.kinds[-1] == lla.POINTER:
            output = builder.alloca(kernel.argtypes[-1].pointee)
        elif not is_scalar: # Array
            kind = kernel.kinds[-1][0]
            eltype = kernel.argtypes[-1].pointee.elements[0].pointee
            assert node.shape is not None
            assert kernel.argtypes[-1].pointee.elements[1].count == len(node.shape)
            output, freefunc, freedata = lla.create_array(
                            builder, node.shape, kind, eltype)
            new = _cleanup(builder, freefunc, freedata)

    #Setup the argument list
    args = [child.llvm_obj for child in node.children]

    if not is_scalar:
        args.append(output)

    # call the kernel corresponding to this node
    # bitcast any arguments that don't match the kernel.function type
    # for array types and pointer types... Needed because inputs might
    # be from different compilers...
    newargs = []
    kfunc_args = kernel.func.type.pointee.args
    for kind, oldarg, needed_type in zip(kernel.kinds, args, kfunc_args):
        newarg = oldarg
        if (kind != lla.SCALAR) and (needed_type != oldarg.type):
            newarg = builder.bitcast(oldarg, needed_type)
        newargs.append(newarg)
    res = builder.call(kernel.func, newargs)
    assert kernel.func.module is builder.basic_block.function.module

    if is_scalar:
        node.llvm_obj = res
    else:
        node.llvm_obj = output

    return new

# This also replaces arguments with the unique argument in the kernel tree
def find_unique_args(tree, unique_args):
    for i, element in enumerate(tree.children):
        if isinstance(element, Argument):
            try:
                index = unique_args.index(element)
            except ValueError: # not found
                unique_args.append(element)
            else:
                tree.children[i] = unique_args[index]
        else:
            find_unique_args(element, unique_args)

def get_fused_type(tree):
    """Get the function type of the compound kernel
    """
    outkrn = tree.kernel
    # If this is not a SCALAR then we need to attach another node
    out_kind = outkrn.kinds[-1]
    out_type = outkrn.func.type.pointee.return_type

    unique_args = []
    find_unique_args(tree, unique_args)

    args = [arg.llvmtype for arg in unique_args]

    if out_kind != lla.SCALAR:
        args.append(outkrn.argtypes[-1])

    return unique_args, lc.Type.function(out_type, args)

def fuse_kerneltree(tree, module_or_name):
    """Fuse the kernel tree into a single kernel object with the common names

    Examples:

    add(multiply(b,c),subtract(d,f))

    var tmp0 = multiply(b,c)
    var tmp1 = subtract(d,f)

    return add(tmp0, tmp1)

    var tmp0;
    var tmp1;

    multiply(b,c,&tmp0)
    subtract(d,f,&tmp1)

    add(tmp0, tmp1, &res)
    """
    if isinstance(module_or_name, py2help._strtypes):
        module = Module.new(module_or_name)
    else:
        module = module_or_name
    args, func_type = get_fused_type(tree)
    outdshape = tree.kernel.dshapes[-1]


    try:
        func = module.get_function_named(tree.name+"_fused")
    except LLVMException:
        func = lc.Function.new(module, func_type, tree.name+"_fused")
        block = func.append_basic_block('entry')
        builder = lc.Builder.new(block)

        # TODO: Create wrapped function for functions
        #   that need to loop over their inputs

        # Attach the llvm_object to the Argument objects
        for i, arg in enumerate(args):
            arg.llvm_obj = func.args[i]

        # topologically sort the kernel-tree nodes and then for each node
        #  site we issue instructions to compute the value
        nodelist = tree.sorted_nodes()


        cleanup = []  # Objects to deallocate any temporary heap memory needed
                      #   ust have a _dealloc method
        def _temp_cleanup():
            for obj in cleanup:
                if obj is not None:
                    obj._dealloc()

        #import pdb
        #pdb.set_trace()

        for node in nodelist[:-1]:
            node.kernel.attach(module)
            new = insert_instructions(node, builder)
            cleanup.append(new)

        nodelist[-1].kernel.attach(module)

        if tree.kernel.kinds[-1] == lla.SCALAR:
            new = insert_instructions(nodelist[-1], builder)
            cleanup.append(new)
            _temp_cleanup()
            builder.ret(nodelist[-1].llvm_obj)
        else:
            new = insert_instructions(nodelist[-1], builder, func.args[-1])
            cleanup.append(new)
            _temp_cleanup()
            builder.ret_void()

    dshapes = [arg.get_kernel_dshape() for arg in args]
    dshapes.append(outdshape)
    newkernel = BlazeElementKernel(func, dshapes)

    return newkernel, args

