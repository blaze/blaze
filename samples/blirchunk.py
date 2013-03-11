import os

import blaze
from blaze.blir import compile, Context, execute

# This is kind of verbose at the moment, idea is to show what
# would be the internals of chunked execution of a BLIR kernel...

#------------------------------------------------------------------------
# Blir Kernel
#------------------------------------------------------------------------

source = """
def main(x : array[int], y : array[int], n : int) -> int {
    var int accum = 0;
    var int i;
    for i in range(n) {
        accum = accum + x[i] * y[i];
    }
    return accum;
}

"""

#------------------------------------------------------------------------
# Load Data
#------------------------------------------------------------------------

ds = blaze.dshape('50000, int32')

if not os.path.exists('x'):
    x = blaze.ones(ds, params=blaze.params(storage='x'))
else:
    x = blaze.open('x')

if not os.path.exists('y'):
    y = blaze.ones(ds, params=blaze.params(storage='y'))
else:
    y = blaze.open('y')

#------------------------------------------------------------------------
# Setup the Iterator
#------------------------------------------------------------------------

xs = y.data.ca
ys = x.data.ca

ast, env = compile(source)
ctx = Context(env)

res = 0

# On-disk chunks
for i in range(xs.nchunks):
    xi = xs.chunks[i][:]
    yi = ys.chunks[i][:]

    size = xi.shape[0]
    args = (xi, yi, size)

    res += execute(ctx, args, fname='main')

# In-memory leftovers
xi = xs.leftover_array
yi = ys.leftover_array
size = xi.shape[0]
args = (xi, yi, size)
res += execute(ctx, args, fname='main')

print 'Result', res

ctx.destroy()
