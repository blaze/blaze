from blaze.blfuncs import BlazeFunc
from blaze.datashape import double, complex128 as c128
import blaze
import array
import ctypes


def _add(a,b):
    return a + b

def _mul(a,b):
    return a * b

print "Begin"

add = BlazeFunc('add',[('f8(f8,f8)', _add),
               ('c16(c16,c16)', _add)])

mul = BlazeFunc('mul', {(double,)*3: _mul,
                           (c128,)*3: _mul})


# Should think about how to generate many of these at once
#  for any source-string approach
#  BLIR or C++
dotcpp = r"""
//#include "stdio.h"

double ddot(Array_C<double, 1> *a, Array_C<double, 1> *b) {
       int i;
       double ret;

       //printf("dims = %ld, %ld\n", a->dims[0], b->dims[0]);
       ret = 0.0;
       for (i=0; i < a->dims[0]; i++) {
           //printf("vals = %f, %f\n", a->data[i], b->data[i]);
           ret += (a->data[i] * b->data[i]);
       }
       return ret;
}
"""

dot = BlazeFunc('dot', [('cpp', dotcpp)])

print "Hello..."

a = blaze.array([1,2,3],dshape=c128)
b = blaze.array([2,3,4],dshape=c128)

print "Everywhere..."

c = add(a,b)
d = mul(c,c)

import ctypes
def _convert(val):
   return val.real + 1j*val.imag
arg1 = blaze.complex128(3.0, 4.0)
arg2 = blaze.complex128(2.0, 5.0)
out = blaze.complex128(0.0, 0.0)
cb = ctypes.byref
d._data.kerneltree(cb(arg1), cb(arg2), cb(out))
out_c = _convert(out)
arg1_c = _convert(arg1)
arg2_c = _convert(arg2)
assert out_c == (arg1_c+arg2_c)**2


af = blaze.array([1,2,3],dshape=double)
bf = blaze.array([2,3,4],dshape=double)

cf = add(af,bf)
df = mul(cf,cf)
# Fuse the BlazeFunc DataDescriptor
# You can call the kerneltree to compute elements (which will fuse the kernel)
ck = df._data.kerneltree.single_ckernel
assert  df._data.kerneltree(3.0, 4.0) == 49.0

result = dot(af, bf)
import array
import ctypes
data = array.array('d',range(10))
def _converta(arr, struct):
    address, count = arr.buffer_info()
    buff = ctypes.cast(address, ctypes.POINTER(ctypes.c_double))
    shape = (ctypes.c_ssize_t * 1)(count)
    val = struct(buff, shape)
    return val
struct = result._data.kerneltree.ctypes_func.argtypes[0]._type_
val = _converta(data, struct)
assert result._data.kerneltree(cb(val), cb(val)) == 285.0

# FIXME:
# Looks like we are still not quite right here...
#  Works for some cases but not for others.
# Adding printf seems to "fix" issues
#  But then repeated calls causes crash
# 
# My guess is we need to look at linking
#  and the function -> module assumption
#  Each BlazeElement Kernel should likely be in it's own module
#  and we make "references" to other named functions
#  Linking happens only during fusion...
#  But, for cases where we lift the kernel --- fusion happens and
#    we need to link a couple of times.
gf = mul(af, bf)
result2 = dot(cf, gf)
ktree = result2._data.kerneltree
struct1 = ktree.ctypes_func.argtypes[0]._type_
struct2 = ktree.ctypes_func.argtypes[1]._type_
data1 = array.array('d',[1,2,3])
data2 = array.array('d',[2,3,4])
val1 = _converta(data1, struct1)
val2 = _converta(data2, struct2)
new = ktree(cb(val1), cb(val2))
print new
gdata = [u*v for u,v in zip(data1, data2)]
cdata = [u+v for u,v in zip(data1, data2)]
actual = sum(x*y for x,y in zip(cdata, gdata))
print actual
