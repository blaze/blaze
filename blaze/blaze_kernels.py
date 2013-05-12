from __future__ import absolute_import

# A Blaze Element Kernel is a wrapper around an LLVM Function with a
#    particular signature.
#    The kinds of argument types are simple, ptr, and array.
#    A kernel kind is a tuple of input kinds followed by the output kind
#
#    simple:  out_type @func(in1_type %a, in2_type %b)
#    ptrs:  void @func(in1_type * %a, in2_type * %b, out_type * %out)
#    array:  void @func(in1_array * %a, in2_array * %b, out_array * %out)
#
# We use a simple array type definition at this level for arrays
# struct {
#    eltype *data;
#    int nd;
#    diminfo dims[nd];
#} array
#
# struct {
#   intp dim;
#   intp stride;
#} diminfo
#

import sys

import llvm.core as lc
from llvm.core import Type, Function, Module
from llvm import LLVMException

void_type = Type.void()
int_type = Type.int()
intp_type = Type.int(8) if sys.maxsize > 2**32 else Type.int(4)
diminfo_type = Type.struct([
                            intp_type,    # shape
                            intp_type     # stride
                            ], name='diminfo')

array_type = lambda el_type, nd: Type.struct([
    Type.pointer(el_type),       # data
    int_type,                    # nd
    Type.array(diminfo_type, nd) # dims[nd]  
                                 # use 0 for a variable-length struct
    ])

generic_array_type = array_type(Type.int(8), 0)

SCALAR = 0
POINTER = 1
ARRAY = 2

arg_kinds = (SCALAR, POINTER, ARRAY)

def isarray(arr):
    if not isinstance(arr, lc.StructType):
        return False
    if arr.element_count != 3 or \
        not isinstance(arr.elements[0], lc.PointerType) or \
        not arr.elements[1] == int_type or \
        not isinstance(arr.elements[2], lc.ArrayType):
        return False
    shapeinfo = arr.elements[2]
    if not shapeinfo.element == diminfo_type:
        return False
    return True

# A wrapper around an LLVM Function object
# Every LLVM Function object comes attached to a particular module
# But, Blaze Element Kernels may be re-attached to different LLVM modules
# as needed using the attach method.
# 
# To inline functions we can either:
#  1) Execute f.add_attribute(lc.ATTR_ALWAYS_INLINE) to always inline a particular
#     function 'f'
#  2) Execute llvm.core.inline_function(callinst) on the output of the 
#     call function when the function is used.
class BlazeElementKernel(object):
    def __init__(self, func):
        if not isinstance(func, Function):
            raise ValueError("Function should be an LLVM Function."\
                                " Try a converter method.")


        func.add_attribute(lc.ATTR_ALWAYS_INLINE)
        self.func = func
        func_type = func.type.pointee
        self.argtypes = func_type.args
        self.return_type = func_type.return_type
        kindlist = [None]*func_type.arg_count
        if not (func_type.return_type == void_type):  # Scalar output
            kindlist += [SCALAR]

        for i, arg in enumerate(func_type.args):
            if not isinstance(arg, lc.PointerType):
                kindlist[i] = SCALAR
            elif isarray(arg.pointee):
                kindlist[i] = ARRAY
                kindlist[i] = None  # unknown
            else:
                kindlist[i] = POINTER
        self.kind = tuple(kindlist)
        # Keep a handle on the module object
        self.module = func.module

    def verify_ranks(self, ranks):
        for i, kind in enumerate(self.kind):
            if (kind == ARRAY or kind is None) and self.rank[0] == 0:
                raise ValueError("Non-scalar function argument "\
                                 "but scalar rank in argument %d" % i)


    @property
    def nin(self):
        return len(self.kind)-1

    # Currently only works for scalar kernels
    @staticmethod
    def frompyfunc(pyfunc, signature):
        import numba
        return BlazeElementKernel(numba.jit(signature)(pyfunc).lfunc)

    @staticmethod
    def fromblir(str):
        raise NotImplementedError

    @staticmethod
    def fromcffi(cffifunc):
        raise NotImplementedError

    @staticmethod
    def fromctypes(func, module=None):
        if func.argtypes is None:
            raise ValueError("ctypes function must have argtypes and restype set")
        if module is None:
            names = [arg.__name__ for arg in func.argtypes]
            names.append(func.restype.__name__)
            name = "mod__{0}_{1}".format(func.__name__, '_'.join(names))
            module = Module.new(name)
        raise NotImplementedError

    @staticmethod
    def fromcfunc(cfunc):
        raise NotImplementedError

    # Should check to ensure kinds still match
    def replace_func(self, func):
        self.func = func

    def attach(self, module):
        """Update this kernel to be attached to a particular module

        Return None
        """

        if not isinstance(module, Module):
            raise TypeError("Must provide an LLVM module object to attach kernel")
        if module is self.module: # Already attached
            return
        try:
            # This assumes unique names for functions and just
            # replaces the function with one named from module
            # Should check to ensure kinds still match
            self.func = module.get_function_named(self.func.name)
            self.module = module
            return
        except LLVMException:
            pass

        # Link the module the function is part of to this module
        module.link_in(self.func.module, preserve=True)
        # Re-set the function object to the newly linked function
        self.func = module.get_function_named(self.func.name)
        self.module = module
       

    def create_wrapper_kernel(input_ranks, output_rank):
        """Take the current kernel and available input argument ranks
         and create a new kernel that matches the required output rank 
         by using the current kernel multiple-times if necessary. 

         This kernel allows creation of a simple call stack.

        Example: (let rn == rank-n) 
          We need an r2, r2 -> r2 kernel and we have an r1, r1 -> r1
          kernel.    

          We create a kernel with rank r2, r2 -> r2 that does the equivalent of

          for i in range(n0):
              out[i] = inner_kernel(in0[i], in1[i])
        """

        raise NotImplementedError

# A Node on the kernel Tree
class KernelObj(object):
    def __init__(self, kernel, ranks, name):
        if not isinstance(kernel, BlazeElementKernel):
            raise ValueError("Must pass in kernel object of type BlazeElementKernel")
        self.kernel = kernel
        self.ranks = ranks
        self.name = name  # name of kernel
        kernel.verify_ranks(ranks)

    def attach_to(self, module):
        """attach the kernel to a different LLVM module
        """
        self.kernel.attach(module)

# An Argument to a kernel tree (encapsulates the array, argument kind and rank)
class Argument(object):
    def __init__(self, arg, kind, rank, llvmtype):
        self.arg = arg
        self.kind = kind
        self.rank = rank
        self.llvmtype = llvmtype

    def __eq__(self, other):
        if not isinstance(other, Argument):
            return NotImplemented
        return (self.arg is other.arg) and (self.rank == other.rank)

# This find args that are used uniquely as well.
def find_unique_args(tree, unique_args):
    for element in tree.children:
        if isinstance(element, Argument):
            if element not in unique_args:
                unique_args.append(element)
        else:
            find_unique_args(element, unique_args)

def get_fused_type(tree):
    """Get the function type of the compound kernel
    """
    outkrn = tree.node.kernel
    # If this is not a SCALAR then we need to attach another node
    out_kind = outkrn.kind[-1]
    out_type = outkrn.func.type.pointee.return_type

    unique_args = []
    find_unique_args(tree, unique_args)

    args = [arg.llvmtype for arg in unique_args]

    if out_kind != SCALAR:
        args.append(outkrn.argtypes[-1])

    return unique_args, Type.function(out_type, args)

# This modifies the node to add a reference to the llvm_obj
def insert_instructions(node, builder):
    kernel = node.node.kernel
    is_scalar = (kernel.kind[-1] == SCALAR)
    #allocate space for output if necessary
    if not is_scalar:
        # FIXME
        print "Adding alloc %s" % kernel.argtypes[-1]
        output = builder.alloca(kernel.argtypes[-1])

    #Setup the argument list
    args = [child.llvm_obj for child in node.children]

    if not is_scalar:
        args.append(output)

    #call the kernel corresponding to this node
    res = builder.call(kernel.func, args, name=node.name)    

    if is_scalar:
        node.llvm_obj = res
    else:
        node.llvm_obj = output


def fuse_kerneltree(tree, newname):
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
    module = Module.new(newname)
    args, func_type = get_fused_type(tree)
    func = Function.new(module, func_type, tree.name+"_fused")
    block = func.append_basic_block('entry')
    builder = lc.Builder.new(block)

    # TODO: Create wrapped function for functions
    #   that need to loop over their inputs

    # Attach the llvm_object to the Argument objects
    for i, arg in enumerate(args):
        arg.llvm_obj = func.args[i]

    # topologically sort the kernel-tree nodes at then for each node 
    #  site we issue instructions to compute the value
    nodelist = tree.sorted_nodes()

    for node in nodelist:
        node.node.attach_to(module)
        insert_instructions(node, builder)

    if tree.node.kernel.kind[-1] == SCALAR:
        builder.ret(nodelist[-1].llvm_obj)
    else:  
        builder.ret_void()

    newkernel = BlazeElementKernel(func)
    ranks = [arg.rank for arg in args]

    krnlobj = KernelObj(newkernel, ranks, newname)

    return krnlobj, args 