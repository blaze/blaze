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
import ctypes
import llvm.core as lc
from llvm.core import Type, Function, Module
from llvm import LLVMException
from .. import llvm_array as lla
from ..llvm_array import (void_type, intp_type,
                SCALAR, POINTER, array_kinds, check_array,
                get_cpp_template, array_type, const_intp, LLArray, orderchar)
from .kernelgen import loop_nest
from ..py2help import izip, _strtypes, c_ssize_t, PY2
from ..datashape import Fixed, TypeVar
from ..datashape.util import to_ctypes, dshape as make_dshape
from .llutil import (int32_type, int8_p_type, single_ckernel_func_type,
                map_llvm_to_ctypes)
from .jit_ckernel import jit_compile_unbound_single_ckernel

arg_kinds = (SCALAR, POINTER) + array_kinds

_g = globals()
for this in array_kinds:
    _g[lla.kind_to_str(this)] = this
del this, _g

_invmap = {}

class BlazeElementKernel(object):
    """
    A wrapper around an LLVM Function object
    But, Blaze Element Kernels may be re-attached to different LLVM
    modules as needed using the attach method.

    To inline functions we can either:
     1) Execute f.add_attribute(lc.ATTR_ALWAYS_INLINE) to always inline
        a particular function 'f'
     2) Execute llvm.core.inline_function(callinst) on the output of the
        call function when the function is used.

    If dshapes is provided then this will be a seq of data-shape objects
     which can be helpful in generating a ctypes-callable wrapper
     otherwise the dshape will be inferred from llvm function (but this
     loses information like the sign).
    """
    _func_ptr = None
    _ctypes_func = None
    _ee = None
    _dshapes = None
    _lifted_cache = {}
    _shape_func = None
    def __init__(self, func, dshapes=None):
        if not isinstance(func, Function):
            raise ValueError("Function should be an LLVM Function."\
                                " Try a converter method.")
        self.func = func
        # We are forcing blaze functions to fully inline
        func.add_attribute(lc.ATTR_ALWAYS_INLINE)

        # Convert the LLVM function type into arguments
        func_type = func.type.pointee
        self.argtypes = func_type.args
        self.return_type = func_type.return_type
        kindlist = [None]*func_type.arg_count
        # The output may either via the function's
        # return, or if the function is void, an
        # extra pointer argument at the end.
        if func_type.return_type != void_type:
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
        self._init_dshapes(dshapes)

    @property
    def kinds(self):
        """An array of 'kinds' describing the parameters the
        kernel function accepts. Each kind may be SCALAR, POINTER,
        or a 3-tuple (array_kind, ndim, llvm_eltype).
        """
        return self._kinds

    def _init_dshapes(self, dshapes):
        if dshapes is None:
            # Create dshapes from llvm if none provided
            from ..datashape.util import from_llvm
            ds = [from_llvm(llvm, kind)
                   for llvm, kind in zip(self.argtypes, self.kinds)]
            if self.kinds[-1] == SCALAR:
                ds.append(from_llvm(self.return_type))
            self._dshapes = tuple(ds)
            self.ranks = [len(el)-1 if el else 0 for el in ds]
        else:
            for i, kind in enumerate(self.kinds):
                 if isinstance(kind, tuple) and kind[0] in array_kinds and \
                          len(dshapes[i]) == 1 and not kind[1]==0:
                    raise ValueError("Non-scalar function argument "
                                     "but scalar rank in argument %d" % i)
            self._dshapes = tuple(dshapes)
            self.ranks = [len(el)-1 for el in dshapes]

    @property
    def dshapes(self):
        return self._dshapes

    def _get_ctypes(self):
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
        """a.shapefunc(*shapes)

        This function maps argument shapes to an output shape,
        using the dshape signature of the blaze kernel. It does
        this by matching the TypeVar shapes in the input datashapes
        which have a corresponding entry in the output datashape.
        """
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

    def make_unbound_ckernel(self, strided):
        return jit_compile_unbound_single_ckernel(self, strided=strided)

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
            from ..llvm_array import kindfromchar
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
def frompyfunc(pyfunc, signature, dshapes=None):
    import numba
    from ..datashape.util import from_numba
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
    if dshapes is None:
        dshapes = [from_numba(arg) for arg in numbafunc.signature.args]
        dshapes.append(from_numba(numbafunc.signature.return_type))
    krnl = BlazeElementKernel(numbafunc.lfunc, dshapes)
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

