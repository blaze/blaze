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

import sys
import operator
import llvm.core as lc
from llvm.core import Type, Function, Module
from llvm import LLVMException
from . import llvm_array as lla
from .llvm_array import (void_type, intp_type, array_kinds, check_array,
                get_cpp_template, array_type, const_intp, LLArray, orderchar)
from .kernelgen import loop_nest
from .ckernel import (ExprSingleOperation, JITKernelData,
                UnboundCKernelFunction)
from .py3help import izip, _strtypes, c_ssize_t
from .datashape import Fixed, TypeVar
from .datashape.util import to_ctypes, dshape as make_dshape

int32_type = Type.int(32)

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
    _unbound_single_ckernel = None
    _ee = None
    _dshapes = None
    _lifted_cache = {}
    def __init__(self, func, dshapes=None, shapefunc=None):
        if not isinstance(func, Function):
            raise ValueError("Function should be an LLVM Function."\
                                " Try a converter method.")

        self._shape_func = shapefunc
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
        self._kinds = tuple(kindlist)
        # Keep a handle on the module object
        self.module = func.module
        if dshapes is None:
            dshapes = self.dshapes
        else:
            self.dshapes = dshapes

    @property
    def kinds(self):
        """An array of 'kinds' describing the parameters the
        kernel function accepts. Each kind may be SCALAR, POINTER,
        or a 3-tuple (array_kind, ndim, llvm_eltype).
        """
        return self._kinds

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
                      len(_dshapes[i]) == 1 and not kind[1]==0:
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
        argtypes = [refresh_name(typ, self.module) for typ in self.argtypes]
        return out_type, map(map_llvm_to_ctypes, argtypes, modules, names)

    @property
    def nin(self):
        return len(self.kind)-1

    @property
    def shapefunc(self):
        if self._shape_func is None:
            if self.ranks[-1] == 0:
                self._shape_func = lambda *args: ()
            else:
                symbols = [[sh.symbol for sh in dshape.shape]
                                for dshape in self.dshapes]
                outshapes = []
                for symbol in symbols[-1]:
                    # Find first occurrence of symbol in other shapes
                    for i, arg in enumerate(symbols[:-1]):
                        try:
                            index = arg.index(symbol)
                            break
                        except ValueError:
                            continue
                    outshapes.append((i, index))
                #outshapes is a list of tuples where first argument is which arg
                #and second is which dim
                def shape_func(*args):
                    shape = tuple(args[i][j] for i,j in outshapes)
                    return shape

                self._shape_func = shape_func

        return self._shape_func



    @staticmethod
    def fromcfunc(cfunc):
        raise NotImplementedError

    @property
    def func_ptr(self):
        if self._func_ptr is None:
            module = self.module.clone()
            if self._ee is None:
                from llvm.passes import build_pass_managers
                import llvm.ee as le
                tm = le.TargetMachine.new(opt=3, cm=le.CM_JITDEFAULT, features='')
                pms = build_pass_managers(tm, opt=3, fpm=False,
                                vectorize=True, loop_vectorize=True)
                pms.pm.run(module)
                if sys.version_info >= (3,):
                    import builtins
                else:
                    import __builtin__ as builtins
                builtins._temp = module.clone()
                builtins._tempname = self.func.name
                #self._ee = le.ExecutionEngine.new(module)
                # FIXME: Temporarily disabling AVX, because of misdetection
                #        in linux VMs. Some code is in llvmpy's workarounds
                #        submodule related to this.
                self._ee = le.EngineBuilder.new(module).mattrs("-avx").create()
            func = module.get_function_named(self.func.name)
            self._func_ptr = self._ee.get_pointer_to_function(func)
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
    def unbound_single_ckernel(self):
        """Creates an UnboundCKernelFunction with the ExprSingleOperation prototype.
        """
        import ctypes
        if self._unbound_single_ckernel is None:
            i8_p_type = Type.pointer(Type.int(8))
            func_type = Type.function(void_type,
                            [i8_p_type, Type.pointer(i8_p_type), i8_p_type])
            module = self.module.clone()
            single_ck_func_name = self.func.name +"_single_ckernel"
            single_ck_func = Function.new(module, func_type,
                                              name=single_ck_func_name)
            block = single_ck_func.append_basic_block('entry')
            builder = lc.Builder.new(block)
            dst_ptr_arg, src_ptr_arr_arg, extra_ptr_arg = single_ck_func.args
            dst_ptr_arg.name = 'dst_ptr'
            src_ptr_arr_arg.name = 'src_ptrs'
            extra_ptr_arg.name = 'extra_ptr'
            # Build up the kernel data structure. Currently, this means
            # adding a shape field for each array argument. First comes
            # the kernel data prefix with a spot for the 'owner' reference added.
            input_field_indices = []
            kernel_data_fields = [Type.struct([i8_p_type]*3)]
            kernel_data_ctypes_fields = [('base', JITKernelData)]
            for i, (kind, a) in enumerate(izip(self.kinds, self.argtypes)):
                if isinstance(kind, tuple):
                    if kind[0] != lla.C_CONTIGUOUS:
                        raise ValueError('only support C contiguous array presently')
                    input_field_indices.append(len(kernel_data_fields))
                    kernel_data_fields.append(Type.array(
                                    intp_type, len(self.dshapes[i])-1))
                    kernel_data_ctypes_fields.append(('operand_%d' % i,
                                    c_ssize_t * (len(self.dshapes[i])-1)))
                elif kind in [SCALAR, POINTER]:
                    input_field_indices.append(None)
                else:
                    raise TypeError(("unbound_single_ckernel codegen doesn't " +
                                    "support the parameter kind %r yet") % (k,))
            # Make an LLVM and ctypes type for the extra data pointer.
            kernel_data_llvmtype = Type.struct(kernel_data_fields)
            class kernel_data_ctypestype(ctypes.Structure):
                _fields_ = kernel_data_ctypes_fields
            # Cast the extra pointer to the right llvm type
            extra_struct = builder.bitcast(extra_ptr_arg,
                            Type.pointer(kernel_data_llvmtype))
            # Convert the src pointer args to the
            # appropriate kinds for the llvm call
            args = []
            for i, (kind, atype) in enumerate(izip(self.kinds[:-1], self.argtypes)):
                if kind == SCALAR:
                    src_ptr = builder.bitcast(builder.load(
                                    builder.gep(src_ptr_arr_arg,
                                            (lc.Constant.int(intp_type, i),))),
                                        Type.pointer(atype))
                    src_val = builder.load(src_ptr)
                    args.append(src_val)
                elif kind == POINTER:
                    src_ptr = builder.bitcast(builder.load(
                                    builder.gep(src_ptr_arr_arg,
                                            (lc.Constant.int(intp_type, i),))),
                                        Type.pointer(atype))                    
                    args.append(src_ptr)
                elif isinstance(kind, tuple):
                    src_ptr = builder.bitcast(builder.load(
                                    builder.gep(src_ptr_arr_arg,
                                            (lc.Constant.int(intp_type, i),))),
                                        Type.pointer(kind[2]))
                    # First get the shape of this parameter. This will
                    # be a combination of Fixed and TypeVar (Var unsupported
                    # here for now)
                    shape = self.dshapes[i][:-1]
                    # Get the llvm array
                    arr_var = builder.alloca(atype.pointee)
                    builder.store(src_ptr,
                                    builder.gep(arr_var,
                                    (lc.Constant.int(int32_type, 0),
                                     lc.Constant.int(int32_type, 0))))
                    for j, sz in enumerate(shape):
                        if isinstance(sz, Fixed):
                            # If the shape is already known at JIT compile time,
                            # insert the constant
                            shape_el_ptr = builder.gep(arr_var,
                                            (lc.Constant.int(int32_type, 0),
                                             lc.Constant.int(int32_type, 1),
                                             lc.Constant.int(intp_type, j)))
                            builder.store(lc.Constant.int(intp_type,
                                                    operator.index(sz)),
                                            shape_el_ptr)
                        elif isinstance(sz, TypeVar):
                            # TypeVar types are only known when the kernel is bound,
                            # so copy it from the extra data pointer
                            sz_from_extra_ptr = builder.gep(extra_struct,
                                            (lc.Constant.int(int32_type, 0),
                                             lc.Constant.int(int32_type,
                                                    input_field_indices[i]),
                                             lc.Constant.int(intp_type, j)))
                            sz_from_extra = builder.load(sz_from_extra_ptr)
                            shape_el_ptr = builder.gep(arr_var,
                                            (lc.Constant.int(int32_type, 0),
                                             lc.Constant.int(int32_type, 1),
                                             lc.Constant.int(intp_type, j)))
                            builder.store(sz_from_extra, shape_el_ptr)
                        else:
                            raise TypeError(("unbound_single_ckernel codegen doesn't " +
                                            "support dimension type %r") % type(sz))
                    args.append(arr_var)
            # Call the function and store in the dst
            kind = self.kinds[-1]
            func = module.get_function_named(self.func.name)
            if kind == SCALAR:
                dst_ptr = builder.bitcast(dst_ptr_arg,
                                Type.pointer(self.return_type))
                dst_val = builder.call(func, args)
                builder.store(dst_val, dst_ptr)
            elif kind == POINTER:
                dst_ptr = builder.bitcast(dst_ptr_arg,
                                Type.pointer(self.return_type))                
                builder.call(func, args + [dst_ptr])
            elif isinstance(kind, tuple):
                dst_ptr = builder.bitcast(dst_ptr_arg,
                                Type.pointer(kind[2]))
                # First get the shape of the output. This will
                # be a combination of Fixed and TypeVar (Var unsupported
                # here for now)
                shape = self.dshapes[-1][:-1]
                # Get the llvm array
                arr_var = builder.alloca(self.argtypes[-1].pointee)
                builder.store(dst_ptr,
                                builder.gep(arr_var,
                                    (lc.Constant.int(int32_type, 0),
                                    lc.Constant.int(int32_type, 0))))
                for j, sz in enumerate(shape):
                    if isinstance(sz, Fixed):
                        # If the shape is already known at JIT compile time,
                        # insert the constant
                        shape_el_ptr = builder.gep(arr_var,
                                        (lc.Constant.int(int32_type, 0),
                                         lc.Constant.int(int32_type, 1),
                                         lc.Constant.int(intp_type, j)))
                        builder.store(lc.Constant.int(intp_type,
                                                operator.index(sz)),
                                        shape_el_ptr)
                    elif isinstance(sz, TypeVar):
                        # TypeVar types are only known when the kernel is bound,
                        # so copy it from the extra data pointer
                        sz_from_extra_ptr = builder.gep(extra_struct,
                                        (lc.Constant.int(int32_type, 0),
                                         lc.Constant.int(int32_type,
                                                input_field_indices[-1]),
                                         lc.Constant.int(intp_type, j)))
                        sz_from_extra = builder.load(sz_from_extra_ptr)
                        shape_el_ptr = builder.gep(arr_var,
                                        (lc.Constant.int(int32_type, 0),
                                         lc.Constant.int(int32_type, 1),
                                         lc.Constant.int(intp_type, j)))
                        builder.store(sz_from_extra, shape_el_ptr)
                    else:
                        raise TypeError(("unbound_single_ckernel codegen doesn't " +
                                        "support dimension type %r") % type(sz))
                builder.call(func, args + [arr_var])
            else:
                raise TypeError(("single_ckernel codegen doesn't " +
                                "support kind %r") % kind)
            builder.ret_void()

            #print("Function before optimization passes:")
            #print(single_ck_func)
            #module.verify()

            import llvm.ee as le
            from llvm.passes import build_pass_managers
            tm = le.TargetMachine.new(opt=3, cm=le.CM_JITDEFAULT, features='')
            pms = build_pass_managers(tm, opt=3, fpm=False,
                            vectorize=True, loop_vectorize=True)
            pms.pm.run(module)

            #print("Function after optimization passes:")
            #print(single_ck_func)

            # DEBUGGING: Verify the module.
            #module.verify()
            # TODO: Cache the EE - the interplay with the func_ptr
            #       was broken, so just avoiding caching for now
            # FIXME: Temporarily disabling AVX, because of misdetection
            #        in linux VMs. Some code is in llvmpy's workarounds
            #        submodule related to this.
            ee = le.EngineBuilder.new(module).mattrs("-avx").create()
            func_ptr = ee.get_pointer_to_function(single_ck_func)
            # Create a function which copies the shape from data
            # descriptors to the extra data struct.
            if len(kernel_data_ctypes_fields) == 1:
                def bind_func(estruct, dst_dd, src_dd_list):
                    pass
            else:
                def bind_func(estruct, dst_dd, src_dd_list):
                    for i, (ds, dd) in enumerate(
                                    izip(self.dshapes, src_dd_list + [dst_dd])):
                        shape = [operator.index(dim)
                                        for dim in dd.dshape[-len(ds):-1]]
                        cshape = getattr(estruct, 'operand_%d' % i)
                        for j, dim_size in enumerate(shape):
                            cshape[j] = dim_size

            self._unbound_single_ckernel = UnboundCKernelFunction(
                            ExprSingleOperation(func_ptr),
                            kernel_data_ctypestype,
                            bind_func,
                            (ee, func_ptr))

        return self._unbound_single_ckernel

    # Should probably check to ensure kinds still match
    def replace_func(self, func):
        self.func = func
        self._ee = None
        self._func_ptr = None
        self._ctypes_func = None
        self._single_ckernel = None
        self.module = func.module

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
            return
        except LLVMException:
            pass

        # Link the module the function is part of to this module
        new_module = self.func.module.clone()
        module.link_in(new_module)

        # Re-set the function object to the newly linked function
        self.replace_func(module.get_function_named(self.func.name))


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

        name = self.func.name + "_lifted_%d_%s" % (outrank, orderchar[outkind])
        try_bk = self._lifted_cache.get(name, None)
        if try_bk is not None:
            return try_bk

        if outkind not in array_kinds[:3]:
            raise ValueError("Invalid kind specified for output: %s" % outkind)

        cur_rank = self.ranks[-1]
        if outrank == cur_rank:
            if not (outrank == 0 and all(x in [SCALAR, POINTER] for x in self.kinds)):
                return self  # no-op

        dr = outrank - cur_rank
        if dr < 0:
            raise ValueError("Output rank (%d) must be greater than current "
                             "rank (%d)" % (outrank, cur_rank))

        if not all((x in [SCALAR, POINTER] or x[0]==outkind) for x in self.kinds):
            raise ValueError("Incompatible kernel arguments for "
                             "lifting: %s" % self.kinds)
        # Replace any None values with difference in ranks
        outranks = [ri + dr for ri in self.ranks]


        func_type = self._lifted_func_type(outranks, outkind)
        func = Function.new(self.module, func_type, name=name)
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
        try_bk = BlazeElementKernel(func, dshapes)
        self._lifted_cache[name] = try_bk
        return try_bk

    def _lifted_func_type(self, outranks, outkind):
        argtypes = self.argtypes[:]
        if self.kinds[-1] == SCALAR:
            argtypes.append(self.return_type)
        args = []
        for rank, argtype, kind in zip(outranks, argtypes, self.kinds):
            eltype = get_eltype(argtype, kind)
            arr_type = array_type(rank, outkind, eltype, self.module)
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
    if isinstance(signature, _strtypes):
        jitter = numba.jit(signature)
    elif isinstance(signature, tuple):
        jitter = numba.jit(restype=signature[1], argtypes=signature[0])
    elif isinstance(signature, list):
        jitter = numba.jit(argtypes=signature)
    else:
        raise ValueError("Signature must be list, tuple, "
                         "or string, not %s" % type(signature))
    numbafunc = jitter(pyfunc)
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

def refresh_name(_llvmtype, module=None):
    if (_llvmtype.kind == lc.TYPE_POINTER and
           _llvmtype.pointee.kind == lc.TYPE_STRUCT and
           _llvmtype.pointee.name == ''):
        res = lla.check_array(_llvmtype.pointee)
        if res is None:
            return _llvmtype
        kindnum, nd, eltype = res
        _llvmtype = lc.Type.pointer(array_type(nd, kindnum, eltype, module))
    return _llvmtype

# An Argument to a kernel tree (encapsulates the array, argument kind and rank)
# FIXME --- perhaps we should have _llvmtype be a SCALAR kind or a string that gets converted
#     to the correct llvmtype when needed.
class Argument(object):
    _shape = None
    def __init__(self, arg, kind, rank, llvmtype):
        self.arr = arg
        if isinstance(kind, tuple):
            kind = kind[0]
        self.kind = kind
        self.rank = rank
        self._llvmtype = llvmtype

    def __eq__(self, other):
        if not isinstance(other, Argument):
            return NotImplemented
        # FIXME: Should remove kind check and cast different kinds
        #        in the generated code.
        return (self.arr is other.arr) and (self.rank == other.rank) and (self.kind==other.kind)

    # FIXME:
    #   Because module linking destroys struct names we need to store
    #   a stringified version of the element type and then
    #   convert as needed...
    @property
    def llvmtype(self):
        self._llvmtype = refresh_name(self._llvmtype)
        return self._llvmtype

    @property
    def shape(self):
        if self._shape is None:
            if self.rank == 0:
                self._shape = ()
            else:
                self._shape = self.arr.dshape.shape[-self.rank:]
        return self._shape

    def lift(self, newrank, newkind, module=None):
        oldtype = get_eltype(self.llvmtype, self.kind)
        newtype = lc.Type.pointer(array_type(newrank, newkind, oldtype, module))
        return Argument(self.arr, newkind, newrank, newtype)

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

    if out_kind != SCALAR:
        args.append(outkrn.argtypes[-1])

    return unique_args, Type.function(out_type, args)

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
    is_scalar = (kernel.kinds[-1] == SCALAR)
    #allocate space for output if necessary
    new = None
    if output is None:
        if kernel.kinds[-1] == POINTER:
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
    # bitcast any arguments that don't match the kernel.function type for array types and
    #  pointer types... Needed because inputs might be from different compilers...
    newargs = []
    for kind, oldarg, needed_type in zip(kernel.kinds, args, kernel.func.type.pointee.args):
        newarg = oldarg
        if (kind != SCALAR) and (needed_type != oldarg.type):
            newarg = builder.bitcast(oldarg, needed_type)
        newargs.append(newarg)
    res = builder.call(kernel.func, newargs)
    assert kernel.func.module is builder.basic_block.function.module

    if is_scalar:
        node.llvm_obj = res
    else:
        node.llvm_obj = output

    return new

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
    if isinstance(module_or_name, _strtypes):
        module = Module.new(module_or_name)
    else:
        module = module_or_name
    args, func_type = get_fused_type(tree)
    outdshape = tree.kernel.dshapes[-1]


    try:
        func = module.get_function_named(tree.name+"_fused")
    except LLVMException:
        func = Function.new(module, func_type, tree.name+"_fused")
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

        if tree.kernel.kinds[-1] == SCALAR:
            new = insert_instructions(nodelist[-1], builder)
            cleanup.append(new)
            _temp_cleanup()
            builder.ret(nodelist[-1].llvm_obj)
        else:
            new = insert_instructions(nodelist[-1], builder, func.args[-1])
            cleanup.append(new)
            _temp_cleanup()
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

