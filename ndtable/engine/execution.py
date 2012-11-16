import numba
import ctypes

from llvm.ee import *
from llvm.core import *
from llvm_cbuilder import *
import llvm_cbuilder.shortnames as Cn

# TODO: proprietary what to do, guess we can always emit the bitcode for
# a couple simple functions
from numbapro.vectorize import Vectorize

import numpy as np
import ndtable.carray as ca

LLVM   = 0 # Pure LLVM dispatch
CTYPES = 1 # Dispatch from Python, invoking C function pointers for calls
NUMPY  = 2 # Mock execution using Numpy, for testing.

#------------------------------------------------------------------------
# Execution
#------------------------------------------------------------------------

# A rough analogoue of CArray's eval function

def eval(plan, engine=LLVM):
    pass

#------------------------------------------------------------------------
# Types
#------------------------------------------------------------------------

# TODO: come up with some sane nameing convention for
# distinguishing between LLVM and C Types... otherwise this just
# gets wicked confusing

#  l_void_p -- LLVM Void Pointer
#  c_void_p -- C Void Pointer

# Or maybe...

# li32
# ci32

c_void_p     = ctypes.c_void_p
py_ssize_t   = np.ctypeslib.c_intp
py_ssize_t_p = ctypes.POINTER(py_ssize_t)
void_star    = ctypes.POINTER(c_void_p)

void_p_3     = ctypes.c_void_p*3
void_p_2     = ctypes.c_void_p*2

l_int32   = Type.int(32)
l_char_p  = Type.pointer(Type.int(8))
l_int32_2 = Type.array(Type.int(32), 2)

i8   = Type.int(8)
i32  = Type.int(32)
i64  = Type.int(64)
i32p = Type.pointer(i32)
i8p  = Type.pointer(i8)
i8pp = Type.pointer(Type.pointer(i8))
i64p = Type.pointer(i64)

def const(n):
    return Constant.int(Type.int(), n)

#------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------

def add(a, b):
    return a+b

def mul(a, b):
    return a*b

def add1(a):
    return a + 2

#------------------------------------------------------------------------
# UFunc Generation
#------------------------------------------------------------------------

def generate(pyfn, signature):
    """
    Generate a pointer to a ufunc with signature:

        void (*)(void** args, py_ssize_t* dimensions, py_ssize_t* steps, void* data)

    """
    # build basic native code ufunc
    bv = Vectorize(pyfn, target='stream', backend='ast')

    dom = signature[0:-1]
    cod = signature[-1]

    bv.add(restype=cod, argtypes=dom)

    functions = bv.build_ufunc_core()
    lfunc, ptr = functions[0]

    return lfunc, ptr

def debug(carr):
    pt = carr.chunks[0].pointer
    vw = carr.chunks[0].viewof
    print pt, vw

#------------------------------------------------------------------------
# C Dispatch Unary
#------------------------------------------------------------------------

def blocked_unary_ufunc(ufn, a, sink):
    data = None

    itemsize = a.dtype.itemsize
    steps = (py_ssize_t * 2)(itemsize, itemsize)

    sink_ptr = sink.ctypes.data
    offset = lambda i: sink_ptr+(a.chunklen*itemsize)*i

    # TODO: ask Francesc if this is a method on the carray
    # already
    leftovers = a.leftovers
    leftover_count = (a.len-(len(a.chunks)*a.chunklen))

    # Full Chunks
    # -----------

    for i, chunk in enumerate(a.chunks):
        args = (c_void_p * 2)(
            (a.chunks[i].pointer),
            offset(i)
        )
        dimensions = (py_ssize_t*1)(a.chunklen)
        ufn(args, dimensions, steps, data)

    # Leftover Chunks
    # ---------------
    args = (c_void_p * 2)( (leftovers), offset(i+1) )

    dimensions = (py_ssize_t * 1)(leftover_count)
    ufn(args, dimensions, steps, data)

#------------------------------------------------------------------------
# C Dispatch Binary
#------------------------------------------------------------------------

def blocked_binary_ufunc(ufn, a, b, sink):
    """
    Apply a ufunc over arguments.
    """
    assert a.shape == b.shape, "Input shapes don't align"

    data = None

    itemsize = a.dtype.itemsize
    steps = (py_ssize_t * 3)(itemsize, itemsize, itemsize)

    sink_ptr = sink.ctypes.data
    offset = lambda i: sink_ptr+(a.chunklen*itemsize)*i

    # TODO: ask Francesc if this is a method on the carray
    # already
    leftovers = a.leftovers
    leftover_count = (a.len-(len(a.chunks)*a.chunklen))

    # Full Chunks
    # -----------

    for i, chunk in enumerate(a.chunks):
        args = (c_void_p * 3)(
            (a.chunks[i].pointer),
            (b.chunks[i].pointer),
            offset(i)
        )
        dimensions = (py_ssize_t*1)(a.chunklen)
        ufn(args, dimensions, steps, data)

    # Leftover Chunks
    # ---------------
    args = (c_void_p * 3)( (leftovers), (leftovers), offset(i+1) )

    dimensions = (py_ssize_t * 1)(leftover_count)
    ufn(args, dimensions, steps, data)

class Blufunc(CDefinition):
    _name_ = 'blufunc'
    _retty_ = Cn.int

    def __init__(self, lfunc):
        self._argtys_ = [
            ('ufn'            , lfunc.type),
            ('data'           , l_char_p),
            ('itemsize'       , l_int32),
            ('chunklen'       , l_int32),
            ('leftovers'      , l_char_p),
            ('leftover_count' , l_int32),
        ]

    def body(self, ufn, data, itemsize, chunklen, leftovers, leftover_count):
        """
        The heart of the blufunc, the chunked execution of
        the ufunc kernel from Numba over the CArray
        structure.
        """
        one = self.constant(Cn.int, 0)

        chunk = self.var(Cn.int, 0, name = 'idx')

        with self.loop() as loop:

            with loop.condition() as setcond:
                setcond(chunk < chunklen)

            with loop.body():
                x = self.builder.alloca(Type.array(i8p,2), 'args')
                args = self.builder.bitcast(x, i8pp)

                dimensions = self.builder.alloca(Type.array(i8p,2), 'dimensions')
                dims = self.builder.gep(dimensions, [const(0), const(0)])
                dims2 = self.builder.bitcast(dims, i64p)

                sink = self.builder.alloca(i8)
                self.builder.call(ufn.value, [args, dims2, dims2, sink])
                chunk += one

        return self.ret(self.constant(Cn.int, 1))

#------------------------------------------------------------------------
# LLVM Unary Dispatch
#------------------------------------------------------------------------

def wrap_func(func, engine, py_module ,rettype):
    from bitey.bind import map_llvm_to_ctypes
    args = func.type.pointee.args
    ret_type = func.type.pointee.return_type
    ret_ctype = map_llvm_to_ctypes(ret_type, py_module)
    args_ctypes = [map_llvm_to_ctypes(arg, py_module) for arg in args]

    functype = ctypes.CFUNCTYPE(rettype, *args_ctypes)
    addr = engine.get_pointer_to_function(func)
    return functype(addr)

def eval_unary():
    # =========================
    # int32 -> int32 -> int32
    lfunc, ptr = generate(add1, (numba.int32, numba.int32))
    # =========================

    module = lfunc.module.clone()

    blufunc = Blufunc(lfunc)(module)
    cexec = CExecutor(module)

    ufn_type = ctypes.CFUNCTYPE(None, void_star, py_ssize_t_p, py_ssize_t_p, c_void_p)
    cblufunc = cexec.get_ctype_function(blufunc,
        None,
        ufn_type,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
    )

    dummy = np.array([]).ctypes.data_as(ctypes.c_char_p)

    cfunc = ufn_type(ptr)
    cblufunc(cfunc, dummy, 0, 0, dummy, 1)

#------------------------------------------------------------------------
# Test
#------------------------------------------------------------------------

def test_binary():
    # =========================
    lfunc, _ = generate(add, (numba.int32, numba.int32, numba.int32))
    # =========================

    module = lfunc.module.clone()
    cexec = CExecutor(module)

    cfunc = cexec.get_ctype_function(lfunc, void_p_3, void_p_3, py_ssize_t_p, c_void_p)

    N = 5000
    t = np.dtype('int')

    a = ca.carray(np.arange(0,N), ca.cparams(shuffle=False, clevel=0))
    b = ca.carray(np.arange(0,N*2,2), ca.cparams(shuffle=False, clevel=0))
    sink = np.zeros(N, t)

    blocked_binary_ufunc(cfunc, a, b, sink)

    print sink[0:20]
    print sink[4092:4114]
    print sink[-20:-1]

    assert not np.any(sink[1:]==0)

def test_unary():
    # =========================
    lfunc, _ = generate(add1, (numba.int32, numba.int32))
    # =========================

    module = lfunc.module.clone()
    cexec = CExecutor(module)

    cfunc = cexec.get_ctype_function(lfunc, void_p_2, void_p_2, py_ssize_t_p, c_void_p)

    N = 5000
    t = np.dtype('int')

    a = ca.carray(np.arange(0,N), ca.cparams(shuffle=False, clevel=0))
    sink = np.zeros(N, t)

    blocked_unary_ufunc(cfunc, a, sink)

    print sink[0:20]
    print sink[4092:4114]
    print sink[-20:-1]

    assert not np.any(sink[1:]==0)

if __name__ == '__main__':
    eval_unary()
    test_binary()
    test_unary()
