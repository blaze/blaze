from __future__ import absolute_import
from __future__ import print_function


# A Blaze Element Kernel is a wrapper around an LLVM Function with a
#    particular signature.
#    The kinds of argument types are simple, ptr, and array.
#    A kernel kind is a tuple of input kinds followed by the output kind
#
#    simple:  out_type @func(in1_type %a, in2_type %b)
#    ptrs:  void @func(in1_type * %a, in2_type * %b, out_type * %out)
#    array_0:  void @func(in1_array0 * %a, in2_array0 * %b, out_array0* %out)
#    array_1:  void @func(in1_array1 * %a, in2_array1 * %b, out_array1* %out)
#    array_2:  void @func(in1_array2 * %a, in2_array2 * %b, out_array2* %out)
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
from .llvm_array import (void_type, intp_type, array_kinds, check_array, get_cpp_template,
                         array_type, const_intp, LLArray)
from .kernelgen import loop_nest
from .ckernel import ExprSingleOperation, wrap_ckernel_func
from .py3help import izip
from .datashape.util import dshape as make_dshape

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
    _single_ckernel = None
    _ee = None
    _dshapes = None
    def __init__(self, func, dshapes=None):
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
        if dshapes is None:
            dshapes = self.dshapes
        else:
            self.dshapes = dshapes

    @property
    def dshapes(self):
        if self._dshapes is None:
            # Create dshapes from llvm if none provided
            from .datashape.util import from_llvm
            ds = [from_llvm(llvm, kind)
                   for llvm, kind in zip(self.argtypes, self.kinds)]
            if self.kinds[-1] == SCALAR:
                ds.append(from_llvm(self.return_type))
            self._dshapes = tuple(ds)
            self.ranks = [len(el)-1 if el else 0 for el in ds]
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
        self.ranks = [len(el)-1 for el in _dshapes]

    def _get_ctypes(self):
        from .blir.bind import map_llvm_to_ctypes
        import sys
        if self.return_type == void_type:
            out_type = None
            indx = slice(None)
        else:
            out_type = map_llvm_to_ctypes(self.return_type)
            indx = slice(None,-1)
        names = []
        for x in self.dshapes[indx]:
            if hasattr(x, 'measure'):
                name = str(x.measure)
            elif hasattr(x, 'name'):
                name = str(x.name)
            else:
                name = None
            names.append(name)
        mod = sys.modules['blaze']
        modules = [mod]*len(names)
        return out_type, map(map_llvm_to_ctypes, self.argtypes, modules, names)


    @property
    def nin(self):
        return len(self.kind)-1


    @staticmethod
    def fromcfunc(cfunc):
        raise NotImplementedError

    @property
    def func_ptr(self):
        if self._func_ptr is None:
            if self._ee is None:
                from llvm.passes import build_pass_managers
                import llvm.ee as le
                module = self.module.clone()                  
                tm = le.TargetMachine.new(opt=3, cm=le.CM_JITDEFAULT, features='')
                pms = build_pass_managers(tm, opt=3, fpm=False)
                pms.pm.run(module)
                self._ee = le.ExecutionEngine.new(module)
            self._func_ptr = self._ee.get_pointer_to_function(self.func)
        return self._func_ptr

    @property
    def ctypes_func(self):
        if self._ctypes_func is None:
            import ctypes
            out_type, argtypes = self._get_ctypes()
            FUNC_TYPE = ctypes.CFUNCTYPE(out_type, *argtypes)
            self._ctypes_func = FUNC_TYPE(self.func_ptr)
        return self._ctypes_func

    @property
    def single_ckernel(self):
        """Creates a CKernel object with prototype ExprSingleOperation
        """
        if self._single_ckernel is None:
            i8_p_type = Type.pointer(Type.int(8))
            func_type = Type.function(void_type,
                            [i8_p_type, Type.pointer(i8_p_type), i8_p_type])
            single_ck_func_name = self.func.name +"_single_ckernel"
            single_ck_func = Function.new(self.module, func_type,
                            name=single_ck_func_name)
            block = single_ck_func.append_basic_block('entry')
            builder = lc.Builder.new(block)
            # Convert the src pointer args to the appropriate kinds for the llvm call
            dst_ptr_arg, src_ptr_arr_arg, extra_ptr_arg = single_ck_func.args
            dst_ptr_arg.name = 'dst_ptr'
            src_ptr_arr_arg.name = 'src_ptrs'
            extra_ptr_arg.name = 'extra_ptr'
            args = []
            for i, (k, a) in enumerate(izip(self.kinds, self.argtypes)):
                if k == SCALAR:
                    src_ptr = builder.bitcast(builder.load(
                                    builder.gep(src_ptr_arr_arg,
                                            (lc.Constant.int(intp_type, i),))),
                                        Type.pointer(a))
                    src_val = builder.load(src_ptr)
                    args.append(src_val)
                elif k == POINTER:
                    src_ptr = builder.bitcast(builder.load(
                                    builder.gep(src_ptr_arr_arg,
                                            (lc.Constant.int(intp_type, i),))), a)
                    args.append(src_ptr)
                else:
                    raise TypeError("single_ckernel codegen doesn't support array types yet")
            # Call the function and store in the dst
            dst_ptr = builder.bitcast(dst_ptr_arg, Type.pointer(self.return_type))
            if self.kinds[-1] == SCALAR:
                dst_val = builder.call(self.func, args)
                builder.store(dst_val, dst_ptr)
            elif self.kinds[-1] == POINTER:
                builder.call(self.func, args + [dst_ptr])
            else:
                raise TypeError("single_ckernel codegen doesn't support array types yet")
            builder.ret_void()
            # JIT compile the function
            if self._ee is None:
                module = self.module.clone()
                # Get the function again from the new module
                single_ck_func = module.get_function_named(single_ck_func_name)
                from llvm.ee import ExecutionEngine
                self._ee = ExecutionEngine.new(module)
            func_ptr = self._ee.get_pointer_to_function(single_ck_func)
            self._single_ckernel = wrap_ckernel_func(
                            ExprSingleOperation(func_ptr), func_ptr)
        return self._single_ckernel

    # Should probably check to ensure kinds still match
    def replace_func(self, func):
        self.func = func
        self._ee = None
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


    def lift(self, outrank, outkind):
        """Take the current kernel and "lift" it so that the output has
         rank given by output_rank and kind given by outkind.

         All arguments will have the same kind as outkind in the
         signature of the lifted kernel and all ranks will be
         adjusted the same amount as output_rank

         This creates a new BlazeElementKernel whose function
         calls the underlying kernel's function multiple times.

        Example: (let rn == rank-n)
          We need an r2, r2 -> r2 kernel and we have an r1, r1 -> r1
          kernel.

          We create a kernel with rank r2, r2 -> r2 that does the equivalent of

          for i in range(n0):
              out[i] = inner_kernel(in0[i], in1[i])
        """
        if outkind in 'CFS':
            from .llvm_array import kindfromchar
            outkind = kindfromchar[outkind]
        if outkind not in array_kinds[:3]:
            raise ValueError("Invalid kind specified for output: %s" % outkind)

        cur_rank = self.ranks[-1]
        if outrank == cur_rank:
            return self  # no-op

        dr = outrank - cur_rank
        if dr < 0:
            raise ValueError("Output rank (%d) must be greater than current "
                             "rank (%d)" % (outrank, cur_rank))

        if not all(x in [SCALAR, POINTER, outkind] for x in self.kinds):
            raise ValueError("Incompatible kernel arguments for "
                             "lifting: %s" % self.kinds)
        # Replace any None values with difference in ranks
        outranks = [ri + dr for ri in self.ranks]


        func_type = self._lifted_func_type(outranks, outkind)
        func = Function.new(self.module, func_type, name=self.func.name +"_lifted")
        block = func.append_basic_block('entry')
        builder = lc.Builder.new(block)

        def ensure_llvm(arg, kind):
            if isinstance(arg, LLArray):
                return arg.array_ptr
            else:
                return arg

        arg_arrays = [LLArray(arg, builder) for arg in func.args]
        begins = [const_intp(0)]*dr
        # This is the shape of the output array
        ends = arg_arrays[-1].shape
        loop_nest_ctx = loop_nest(builder, begins, ends)

        with loop_nest_ctx as loop:
            if self.kinds[-1] == SCALAR:
                inargs = arg_arrays[:-1]
                inkinds = self.kinds[:-1]
            else:
                inargs = arg_arrays
                inkinds = self.kinds
            callargs = [ensure_llvm(arg[loop.indices], kind) 
                                for arg, kind in zip(inargs, inkinds)]
            res = builder.call(self.func, callargs)
            if self.kinds[-1] == SCALAR:
                arg_arrays[-1][loop.indices] = res
            builder.branch(loop.incr)

        builder.branch(loop.entry)
        builder.position_at_end(loop.end)
        builder.ret_void()

        def add_rank(dshape, dr):
            new = ["L%d, " % i for i in range(dr)]
            new.append(str(dshape))
            return make_dshape("".join(new))

        dshapes = [add_rank(dshape, dr) for dshape in self.dshapes]
        return BlazeElementKernel(func, dshapes)

    def _lifted_func_type(self, outranks, outkind):
        argtypes = self.argtypes
        if self.kinds[-1] == SCALAR:
            argtypes.append(self.return_type)
        args = []
        for rank, argtype, kind in zip(outranks, argtypes, self.kinds):
            eltype = get_eltype(argtype, kind)
            arr_type = array_type(rank, outkind, eltype)
            args.append(Type.pointer(arr_type))
        return Type.function(void_type, args)

def get_eltype(argtype, kind):
    if kind == SCALAR:
        return argtype
    elif kind == POINTER:
        return argtype.pointee
    else: # Array
        return argtype.pointee.elements[0].pointee

# Currently only works for scalar kernels
def frompyfunc(pyfunc, signature):
    import numba
    from .datashape.util import from_numba
    numbafunc = numba.jit(signature)(pyfunc)
    dshapes = [from_numba(arg) for arg in numbafunc.signature.args]
    dshapes.append(from_numba(numbafunc.signature.return_type))
    krnl = BlazeElementKernel(numbafunc.lfunc, dshapes)
    return krnl

def fromblir(codestr):
    from .datashape.util import from_blir
    ast, env = compile(codestr)
    NUM = 1
    func = env['functions'][NUM]
    RET, ARGS = 1, 2
    dshapes = [from_blir(arg) for arg in func[ARGS]]
    dshapes.append(from_blir(func[RET]))
    krnl = BlazeElementKernel(env['lfunctions'][NUM], dshapes)
    return krnl

def fromcffi(cffifunc):
    raise NotImplementedError

def fromcpp(source):
    import tempfile, os, subprocess
    header = get_cpp_template()
    fid_cpp, name_cpp = tempfile.mkstemp(suffix='.cpp', text=True)
    os.write(fid_cpp, header + source)
    os.close(fid_cpp)
    args = ['clang','-S','-emit-llvm','-O3','-o','-',name_cpp]
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE)
    assembly, err = p1.communicate()
    if err:
        raise RuntimeError("Error trying to compile", err)
    os.remove(name_cpp)
    llvm_module = Module.from_assembly(assembly)

    # Always get the first function ---
    #    assume it is source
    # FIXME:  We could improve this with an independent
    #    parser of the source file
    func = llvm_module.functions[0]
    krnl = BlazeElementKernel(func)
    # Use default llvm dshapes --
    #  could improve this with a parser of source
    return krnl

def fromctypes(func, module=None):
    if func.argtypes is None:
        raise ValueError("ctypes function must have argtypes and restype set")
    if module is None:
        names = [arg.__name__ for arg in func.argtypes]
        names.append(func.restype.__name__)
        name = "mod__{0}_{1}".format(func.__name__, '_'.join(names))
        module = Module.new(name)
    raise NotImplementedError


# An Argument to a kernel tree (encapsulates the array, argument kind and rank)
class Argument(object):
    def __init__(self, arg, kind, rank, llvmtype):
        self.arr = arg
        self.kind = kind
        self.rank = rank
        self.llvmtype = llvmtype

    def __eq__(self, other):
        if not isinstance(other, Argument):
            return NotImplemented
        return (self.arr is other.arr) and (self.rank == other.rank)

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
def insert_instructions(node, builder, output=None):
    kernel = node.kernel
    is_scalar = (kernel.kinds[-1] == SCALAR)
    #allocate space for output if necessary
    if output is None:
        if not is_scalar: # FIXME --- add array handling
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
    outdshape = tree.kernel.dshapes[-1]

    # TODO: Create wrapped function for functions
    #   that need to loop over their inputs

    # Attach the llvm_object to the Argument objects
    for i, arg in enumerate(args):
        arg.llvm_obj = func.args[i]

    # topologically sort the kernel-tree nodes and then for each node
    #  site we issue instructions to compute the value
    nodelist = tree.sorted_nodes()

    for node in nodelist[:-1]:
        node.kernel.attach(module)
        insert_instructions(node, builder)

    if tree.kernel.kinds[-1] == SCALAR:
        insert_instructions(nodelist[-1], builder)
        builder.ret(nodelist[-1].llvm_obj)
    else:
        insert_instructions(nodelist[-1], builder, func.args[-1])
        builder.ret_void()

    dshapes = [get_kernel_dshape(arg) for arg in args]
    dshapes.append(outdshape)
    newkernel = BlazeElementKernel(func, dshapes)

    return newkernel, args

# Given an Argument object, return the correspoding kernel data-shape
def get_kernel_dshape(arg):
    rank = arg.rank
    total_dshape = arg.arr.dshape
    sub = len(total_dshape)-1-rank
    return total_dshape.subarray(sub)

