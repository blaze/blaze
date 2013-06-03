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

mine = BlazeElementKernel(create_func(mod))
print mine.func
res = mine.lift(1, 'C')
cfunc = res.ctypes_func

import array
import ctypes
cb = ctypes.byref
data = array.array('d',range(10000))
odata = array.array('d', [0]*10000)
struct = cfunc.argtypes[0]._type_

def _convert(arr, struct):
    address, count = arr.buffer_info()
    buff = ctypes.cast(address, ctypes.POINTER(ctypes.c_double))
    shape = (ctypes.c_long * 1)(count)
    val = struct(buff, shape)
    return val

val = _convert(data, struct)
out = _convert(odata, struct)


cfunc(cb(val), cb(val), cb(out))

print odata[-10:]