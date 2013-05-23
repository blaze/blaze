from blaze.blfuncs import BlazeFunc
from blaze.datashape import double, complex128 as c128
from blaze.blaze_kernels import BlazeElementKernel
import blaze

def _add(a,b):
    return a + b

def _mul(a,b):
    return a * b

add = BlazeFunc('add',[('f8(f8,f8)', _add),
                       ('c16(c16,c16)', _add)])

mul = BlazeFunc('mul', {(double,)*3: _mul,
                        (c128,)*3: _mul})

a = blaze.array([1,2,3],dshape=c128)
b = blaze.array([2,3,4],dshape=c128)

c = add(a,b)
d = mul(c,c)
d._data = d._data.fuse()


af = blaze.array([1,2,3],dshape=double)
bf = blaze.array([2,3,4],dshape=double)

cf = add(af,bf)
df = mul(cf,cf)
df._data = df._data.fuse()


