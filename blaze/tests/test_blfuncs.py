from blaze.blfuncs import BlazeFunc
from blaze.datashape import double, complex128 as c128
from blaze.blaze_kernels import BlazeElementKernel
import blaze

def _add(a,b):
    return a + b

def _mul(a,b):
    return a * b

krnl = BlazeElementKernel.frompyfunc(_add, 'f8(f8,f8)')
krnl2 = BlazeElementKernel.frompyfunc(_add, 'c16(c16,c16)')
add = BlazeFunc('add', ['','',''],{(double, double, double): krnl,
                                   (c128, c128, c128): krnl2})

krnl = BlazeElementKernel.frompyfunc(_mul, 'f8(f8,f8)')
mul = BlazeFunc('mul', ['']*3, {(double,)*3: krnl})

a = blaze.array([1,2,3],dshape=double)
b = blaze.array([2,3,4],dshape=double)

c = add(a,b)
d = mul(c,c)
d._data = d._data.fuse()



