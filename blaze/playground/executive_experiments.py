from __future__ import print_function

"""Experimentation with the executive

This is done at a data-descriptor level
"""


import blaze
import random
import math
import time
from itertools import product as it_product
from blaze.executive import simple_execute_write



def eval_in_mem(arr, iter_dims, chunk=1):
    res = blaze.empty(arr.dshape)
    simple_execute_write(arr._data, res._data,
                         iter_dims=iter_dims, chunk=chunk)
    return res


if __name__ == '__main__':
    operand_datashape = blaze.dshape('10, 10, 10, float64')

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
    result = eval_in_mem(expr, None)
    print("evaluation-in-mem complete took %f seconds"
          % (time.time()-t))
    print(result)

    t = time.time()
    result = eval_in_mem(expr, 0)
    print("evaluation-in-mem iter_dims=0 took %f seconds"
          % (time.time()-t))
    print(result)

    t = time.time()
    result = eval_in_mem(expr, iter_dims=1)
    print("evaluation-in-mem iter_dims=1 took %f seconds"
          % (time.time()-t))
    print(result)

    t = time.time()
    result = eval_in_mem(expr, iter_dims=1, chunk=4)
    print("evaluation-in-mem iter_dims=1 chunksize=4 took %f seconds"
          % (time.time()-t))
    print(result)

    stor = blaze.Storage('blz://persisted.blz')
    t = time.time()
    result = blaze.eval(expr, storage=stor)
    print("evaluation blz took %f seconds" % (time.time()-t))
    print(result)
    blaze.drop(stor)

    t = time.time()
    result = blaze.eval(expr)
    print("evaluation hierarchical %f seconds" % (time.time()-t))

