from blaze.blfuncs import BlazeFuncDeprecated
from blaze.datashape import double, complex128 as c128, int8
import blaze
import array
import ctypes


def _add(a,b):
    return a + b

def _mul(a,b):
    return a * b

print "Begin"

add = BlazeFuncDeprecated('add')
add.add_template(_add)

mul = BlazeFuncDeprecated('mul', {(double, double, double):_mul})
mul.add_template(_mul)


af = blaze.array([1,2,3,4,5],dshape=int8)
bf = blaze.array([2,3,4,5,6],dshape=int8)

cf = add(af,bf)
df = mul(cf,cf)
# Fuse the BlazeFuncDeprecated DataDescriptor
# You can call the kerneltree to compute elements (which will fuse the kernel)
#ck = df._data.kerneltree.single_ckernel
assert  df._data.kerneltree(3, 4) == 49

