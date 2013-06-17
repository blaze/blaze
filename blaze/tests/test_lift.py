import unittest
import sys
import ctypes
import array
from blaze.blaze_kernels import BlazeElementKernel
import llvm.core as lc
import llvm
from ..py3help import c_ssize_t

mod = lc.Module.new('simple')

def create_func(mod):
    try:
        func = mod.get_function_named('add')
        return func
    except llvm.LLVMException:
        pass
    double = lc.Type.double()
    func_type = lc.Type.function(double, [double, double])
    func = lc.Function.new(mod, func_type, name='add')
    block = func.append_basic_block('entry')
    builder = lc.Builder.new(block)
    val = builder.fadd(func.args[0], func.args[1])
    builder.ret(val)
    return func


class TestKernelLift(unittest.TestCase):
    def test_basic_scalar_lift(self):
        addkernel = BlazeElementKernel(create_func(mod))
        res = addkernel.lift(1, 'C')
        cfunc = res.ctypes_func
        cb = ctypes.byref
        data = array.array('d',range(100))
        odata = array.array('d', [0]*100)
        struct = cfunc.argtypes[0]._type_

        def _convert(arr):
            address, count = arr.buffer_info()
            buff = ctypes.cast(address, ctypes.POINTER(ctypes.c_double))
            shape = (c_ssize_t * 1)(count)
            val = struct(buff, shape)
            return val

        val = _convert(data)
        out = _convert(odata)
        cfunc(cb(val), cb(val), cb(out))
        assert all((outd == ind + ind) for ind, outd in zip(data, odata))

    def test_scalar_lift_to2d(self):
        addkernel = BlazeElementKernel(create_func(mod))
        res = addkernel.lift(2, 'C')
        cfunc = res.ctypes_func
        cb = ctypes.byref
        data = array.array('d',range(150))
        odata = array.array('d', [0]*150)
        struct = cfunc.argtypes[0]._type_

        def _convert(arr):
            address, count = arr.buffer_info()
            buff = ctypes.cast(address, ctypes.POINTER(ctypes.c_double))
            shape = (c_ssize_t * 2)(10, count // 10)
            val = struct(buff, shape)
            return val

        val = _convert(data)
        out = _convert(odata)
        cfunc(cb(val), cb(val), cb(out))
        assert all((outd == ind + ind) for ind, outd in zip(data, odata))


if __name__ == '__main__':
    unittest.main()