import unittest
import sys
import ctypes
import array
from blaze.blaze_kernels import BlazeElementKernel
import llvm.core as lc

mod = lc.Module.new('simple')

def create_func(mod):
    double = lc.Type.double()
    func_type = lc.Type.function(double, [double, double])
    func = lc.Function.new(mod, func_type, name='add')
    block = func.append_basic_block('entry')
    builder = lc.Builder.new(block)
    val = builder.fadd(func.args[0], func.args[1])
    builder.ret(val)
    return func


class TestKernelLift(unittest.TestCase):
    def test_basic_lift(self):
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
            shape = (ctypes.c_ssize_t * 1)(count)
            val = struct(buff, shape)
            return val

        val = _convert(data)
        out = _convert(odata)
        cfunc(cb(val), cb(val), cb(out))
        print out
        assert all((outd == ind + ind) for ind, outd in zip(data, odata))

if __name__ == '__main__':
    unittest.main()