from __future__ import print_function

"""Experimentation with the executive

This is done at a data-descriptor level
"""


import blaze
import random
import math
import time
from itertools import product as it_product

def running_simple_executor():
    pass




if __name__ == '__main__':
    operand_datashape = blaze.dshape('1000, 100, 4, float64')

    op0 = blaze.empty(operand_datashape)
    op1 = blaze.empty(operand_datashape)

    shape = (a.val for a in operand_datashape.shape)

    t = time.time()
    for i in it_product(*[xrange(i) for i in shape]):
        val = random.uniform(-math.pi, math.pi)
        op0[i] = math.sin(val)
        op1[i] = math.cos(val)

    print("initialization took %f seconds" % (time.time()-t))

    expr = op0*op0 + op1*op1

    t = time.time()
    result = blaze.eval(expr)
    print("evaluation took %f seconds" % (time.time()-t))

    t = time.time()
    result = blaze.eval(expr,
                        storage=blaze.Storage('blz://persisted.blz'))
    print("evalutation2 took %f seconds" % (time.time()-t))
