from blaze.blfuncs import BlazeFunc
from blaze.datashape import double, complex128 as c128
import blaze

def _add(a,b):
    return a + b

def _mul(a,b):
    return a * b


add = BlazeFunc('add',[('f8(f8,f8)', _add),
                       ('c16(c16,c16)', _add)])

mul = BlazeFunc('mul', {(double,)*3: _mul,
                        (c128,)*3: _mul})


# Should think about how to generate many of these at once
#  for any source-string approach
#  BLIR or C++
dotcpp = """
double ddot(Array_C<double, 1> *a, Array_C<double, 1> *b) {
       int i;
       double ret;

       for (i=0; i < a->dims[0]; i++) {
           ret += a->data[i] * b->data[i];
       }
       return ret;
}
"""

dot = BlazeFunc('dot', [('cpp', dotcpp)])

a = blaze.array([1,2,3],dshape=c128)
b = blaze.array([2,3,4],dshape=c128)

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
address, count = data.buffer_info()
buff = ctypes.cast(address, ctypes.POINTER(ctypes.c_double))
shape = (ctypes.c_long * 1)(count)
struct = result._data.kerneltree.ctypes_func.argtypes[0]._type_
val = struct(buff, shape)
print result._data.kerneltree(cb(val), cb(val))
print 285.0


