from __future__ import absolute_import
from __future__ import print_function


# A Blaze Element Kernel is a wrapper around an LLVM Function with a
#    particular signature.
#    The kinds of argument types are simple, ptr, and array.
#    A kernel kind is a tuple of input kinds followed by the output kind
#
#    simple:  out_type @func(in1_type %a, in2_type %b)
#    ptrs:  void @func(in1_type * %a, in2_type * %b, out_type * %out)
#    array_0:  void @func(in1_array * %a, in2_array * %b, out_array * %out)
#    array_1:  void @func(in1_array * %a, in2_array * %b, out_array * %out)
#    array_2:  void @func(in1_array * %a, in2_array * %b, out_array * %out)
#
#   where array_n is one of the array_kinds in llvm_array
#   
#   Notice that while the examples have functions with all the
#   same kinds, the kind is a per-argument notion and thus
#   can be mixed and matched.

import llvm.core as lc
from llvm.core import Type, Function, Module
from llvm import LLVMException
from . import llvm_array as lla
from .llvm_array import void_type, array_kinds, check_array

SCALAR = 0
POINTER = 1

arg_kinds = (SCALAR, POINTER) + array_kinds

_g = globals()
for this in array_kinds:
    _g[lla.kind_to_str(this)] = this
del this, _g

_invmap = {}

def kind_to_str(kind):
    global _invmap    
    if not _invmap:
        for key, value in globals().items():
            if isinstance(value, int) and value in arg_kinds:
                _invmap[value] = key
    return _invmap[kind]

def str_to_kind(str):
    trial = eval(str)
    if trial not in arg_kinds:
        raise ValueError("Invalid Argument Kind")
    return eval(str)

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

# If dshapes is provided then this will be a seq of data-shape objects
#  which can be helpful in generating a ctypes-callable wrapper
#  otherwise the dshape will be inferred from llvm function (but this loses
#   information like the sign).
class BlazeElementKernel(object):
    _func_ptr = None
    _ctypes_func = None
    _ee = None
    _dshapes = None
    def __init__(self, func, ranks=None):
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
            kindlist.append(SCALAR)

        for i, arg in enumerate(func_type.args):
            if not isinstance(arg, lc.PointerType):
                kindlist[i] = SCALAR
            else: # kind is a tuple if an array
                kind = check_array(arg.pointee)
                if kind is None:
                    kind = POINTER
                kindlist[i] = kind
        self.kinds = tuple(kindlist)
        # Keep a handle on the module object
        self.module = func.module
        if ranks is None:
            self.ranks = len(kindlist)*[0]

    @property
    def dshapes(self):
        if self._dshapes is None:
            # Create dshapes from llvm if none provided
            from .datashape.util import from_llvm
            ds = [from_llvm(llvm, kind)
                   for llvm, kind in zip(self.argtypes, self.kinds)]
            if self.kinds[-1] == SCALAR:
                ds.append(from_llvm(self.return_type))
            self._dshapes = ds
        return self._dshapes

    # FIXME:  Needs more verification...
    @dshapes.setter
    def dshapes(self, _dshapes):
        for i, kind in enumerate(self.kinds):
             if isinstance(kind, tuple) and kind[0] in array_kinds and \
                      len(_dshapes[i]) == 1:
                raise ValueError("Non-scalar function argument "
                                 "but scalar rank in argument %d" % i)
        self._dshapes = tuple(_dshapes)
        return

    def _get_ctypes(self):
        from .datashape.util import to_ctypes
        if self.return_type == void_type:
            out_type = None
        else:
            out_type = to_ctypes(self.dshapes[-1])
        return out_type, map(to_ctypes, self.dshapes[:-1])

    def _get_wrapper(self, raw_func_ptr):
        raise NotImplementedError
        import ctypes
        from .datashape.util import to_ctypes    
        import numpy as np    
        
        argtypes = self.argtypes
        raw_dshape = [to_ctypes(dshape) for dshape in self.dshapes]
        cfunc_argtypes = []
        transform = []
        func_at_runtime = False
        for i, (kind, dshape) in enumerate(zip(self.kinds, self.dshapes)):
            if kind == SCALAR:
                cfunc_argtypes.append(to_ctypes(dshape))
                transform.append(None)
            elif kind == POINTER:
                ctyp = to_ctypes(dshape)
                cfunc_argtypes.append(ctypes.POINTER(ctyp))
                if issubclass(ctyp, ctypes.Structure):
                    func = lambda x: ctypes.pointer(ctyp(*x))
                else:
                    func = lambda x: ctypes.pointer(ctyp(x))
                transform.append(func)
            else: # array kind -- must complete at run-time
                cfunc_argtypes.append(to_ctypes(dshape.measure))
                transform.append(lambda x: np.asarray(x))

        #Output
        kind = self.kinds[-1]
        dshape = self.dshapes[-1]

        if not func_at_runtime:
            FUNC_TYPE = ctypes.CFUNCTYPE(out_type, *argtypes)
            ctypes_func = FUNC_TYPE(ptr_to_func)

        def func(*args):
            new_args = [transform(arg) for transform, arg in zip(transforms,args)]

            if not _out_scalar:
                pass

            for i, arg in enumerate(args):
                if self.kinds[i] == POINTER:
                    new_arg = ctypes.pointer(argtypes[i])
                elif isinstance(self.kinds[i], tuple):
                    pass

            res = raw_func(*new_args)
            if self.kinds[-1] == SCALAR:
                return res
            elif self.kinds[-1] == POINTER:
                new_args[-1]

        return func

    @property
    def nin(self):
        return len(self.kind)-1

    # Currently only works for scalar kernels
    @staticmethod
    def frompyfunc(pyfunc, signature):
        import numba
        numbafunc = numba.jit(signature)(pyfunc)
        krnl = BlazeElementKernel(numbafunc.lfunc)
        from .datashape.util import from_numba
        dshapes = [from_numba(arg) for arg in numbafunc.signature.args]
        dshapes.append(from_numba(numbafunc.signature.return_type))
        krnl.dshapes = dshapes
        return krnl


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

    @property
    def func_ptr(self):
        if self._func_ptr is None:
            if self._ee is None:
                from llvm.ee import ExecutionEngine
                self._ee = ExecutionEngine.new(self.module)            
            self._func_ptr = self._ee.get_pointer_to_function(self.func)
        return self._func_ptr

    @property
    def ctypes_func(self):
        if self._ctypes_func is None:
            import ctypes
            ptr_to_func = self.func_ptr
            if not all(kind == SCALAR for kind in self.kinds):
                self._ctypes_func = self._get_wrapper(ptr_to_func)
            else:
                out_type, argtypes = self._get_ctypes()
                FUNC_TYPE = ctypes.CFUNCTYPE(out_type, *argtypes)
                self._ctypes_func = FUNC_TYPE(ptr_to_func)
        return self._ctypes_func

    # Should check to ensure kinds still match
    def replace_func(self, func):
        self.func = func
        self._func_ptr = None
        self._ctypes_func = None

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
            self.replace_func(module.get_function_named(self.func.name))
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
    outkrn = tree.kernel
    # If this is not a SCALAR then we need to attach another node
    out_kind = outkrn.kinds[-1]
    out_type = outkrn.func.type.pointee.return_type

    unique_args = []
    find_unique_args(tree, unique_args)

    args = [arg.llvmtype for arg in unique_args]

    if out_kind != SCALAR:
        args.append(outkrn.argtypes[-1])

    return unique_args, Type.function(out_type, args)

# This modifies the node to add a reference to the llvm_obj
def insert_instructions(node, builder):
    kernel = node.kernel
    is_scalar = (kernel.kinds[-1] == SCALAR)
    #allocate space for output if necessary
    if not is_scalar:
        output = builder.alloca(kernel.argtypes[-1].pointee)


    #Setup the argument list
    args = [child.llvm_obj for child in node.children]

    if not is_scalar:
        args.append(output)

    #call the kernel corresponding to this node
    res = builder.call(kernel.func, args)

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
        node.attach_to(module)
        insert_instructions(node, builder)

    if tree.kernel.kinds[-1] == SCALAR:
        builder.ret(nodelist[-1].llvm_obj)
    else:
        builder.ret_void()

    ranks = [arg.rank for arg in args]
    newkernel = BlazeElementKernel(func, ranks)

    return newkernel, args
