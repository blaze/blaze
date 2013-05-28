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
# Fuse the BlazeFunc DataDescriptor
d._data = d._data.fuse()

af = blaze.array([1,2,3],dshape=double)
bf = blaze.array([2,3,4],dshape=double)

cf = add(af,bf)
df = mul(cf,cf)
# Fuse the BlazeFunc DataDescriptor
# You can call the kerneltree to compute elements (which will fuse the kernel)
assert  df._data.kerneltree(3.0, 4.0) == 49.0

result = dot(af, bf)
#result._data.kerneltree

