import numpy as np

from blaze.cgen.blirgen import *

from blaze.blir import compile, assembly, bitcode, Context, execute
from blaze.cgen.fusion import *
from blaze.cgen.utils import namesupply

def test_cgen2():
    with namesupply():

        krn = ElementwiseKernel(
            [
                (IN  , VectorArg((25,), 'array[float]')),
                (IN  , VectorArg((25,), 'array[float]')),
                (OUT , VectorArg((25,), 'array[float]')),
            ],
            '_out0[i0] = exp(_in0[i0] + _in1[i0])',
        )

        krn.verify()
        ast, env = krn.compile()

        ctx = Context(env)

        a = np.array(xrange(25), dtype='double')
        b = np.array(xrange(25), dtype='double')
        c = np.empty_like(b)

        execute(ctx, args=(a,b,c), fname='kernel0', timing=False)
        assert np.allclose(c, np.exp(a + b))
