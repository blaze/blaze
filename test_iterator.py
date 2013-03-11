import ctypes
import numpy as np
from blaze.blir import compile, Context, execute

from blaze import open, ones
from blaze import carray as ca
from blaze.rts.wrapper import Runtime, Source, Sink, view

source = """
def main(c: array[int], n : int) -> void {
    var int i;
    for i in range(n) {
        c[i] = c[i] + 1;
    }
}

def awk(c: array[int], n : int) -> void {
    var int i;
    for i in range(n) {
        c[i] = 510;
    }
}

"""

def chunkwise_kernel():
    ast, env = compile(source)

    #Array = ca.carray(xrange(25000), rootdir='example1', mode='w',
                #dtype='int32', cparams=ca.cparams(clevel=0))
    Array = open('example1', mode='w')
    c = Array.data.ca
    ctx = Context(env)

    for i in range(c.nchunks):
        chunk = c.chunks[i]
        # read only access
        #x = c.chunks[0][:]
        # write access
        x = view(chunk)

        size = x.strides[0]
        args = (x, size)
        execute(ctx, args, fname='main')

        # this does a _save() behind the scenes
        c.chunks[i] = chunk

    ctx.destroy()

    rts = Runtime(1,2,3)
    rts.join()

    print Array

chunkwise_kernel()

"""

x = bl.Array(xrange(5000), dtype='float32')
y = bl.Array(xrange(5000), dtype='float32')

a = 1.0
b = 1.0
c = 1.0
d = 1.0

big = (a*x + b*y).dot(c*x+d*y)
big.eval()

"""

#------------------------------------------------------------------------

"""
def gemm(x: array[float], y : array[float], n : int, a : float, b : float, c : array[float]) -> float {
    var int i;
    for i in range(n) {
        c[i] = a*x[i] + b*y[i];
    }
}
"""

#------------------------------------------------------------------------

"""

def dot(a: array[float], b : array[float], n : int) -> float {
    var int i;
    var int accum;
    for i in range(n) {
        accum = accum + a[i]*b[i];
    }
    return accum;
}

"""
