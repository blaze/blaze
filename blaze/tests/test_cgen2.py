import numpy as np

from blaze.cgen.blirgen import *

from blaze.blir import compile, assembly, bitcode, Context, execute
from blaze.cgen.kernels import *
from blaze.cgen.utils import namesupply

#------------------------------------------------------------------------
# Code Generation ( Level 2 )
#------------------------------------------------------------------------

def test_cgen2_expadd():
    with namesupply():

        krn = ElementwiseKernel(
            [
                (IN  , VectorArg((300,), 'array[float]')),
                (IN  , VectorArg((300,), 'array[float]')),
                (OUT , VectorArg((300,), 'array[float]')),
            ],
            '_out0[i0] = exp(_in0[i0] + _in1[i0])',
        )

        krn.verify()
        ast, env = krn.compile()

        ctx = Context(env)

        a = np.array(xrange(300), dtype='double')
        b = np.array(xrange(300), dtype='double')
        c = np.empty_like(b)

        execute(ctx, args=(a,b,c), fname='kernel0', timing=False)
        assert np.allclose(c, np.exp(a + b))

def test_cgen2_add():
    with namesupply():

        krn = ElementwiseKernel(
            [
                (IN  , VectorArg((300,), 'array[int]')),
                (IN  , VectorArg((300,), 'array[int]')),
                (OUT , VectorArg((300,), 'array[int]')),
            ],
            '_out0[i0] = _in0[i0] + _in1[i0]',
        )

        krn.verify()
        ast, env = krn.compile()

        ctx = Context(env)

        a = np.array(xrange(300), dtype='int32')
        b = np.array(xrange(300), dtype='int32')
        c = np.empty_like(b)

        execute(ctx, args=(a,b,c), fname='kernel0', timing=False)
        assert np.allclose(c, a + b)
