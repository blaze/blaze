from __future__ import print_function

"""Experimentation with the executive

This is done at a data-descriptor level
"""


import blaze
import random
import math
import time
import logging

from itertools import product as it_product
from blaze.executive import simple_execute_write



def eval_in_mem(arr, iter_dims, chunk=1, dump_result=False):
    logging.info("starting in-mem with dims=%s chunk=%d",
                 str(iter_dims), chunk)
    t = time.time()
    res = blaze.zeros(arr.dshape)
    simple_execute_write(arr._data, res._data,
                         iter_dims=iter_dims, chunk=chunk)
    logging.info("took %f secs.\n", time.time()-t);
    if dump_result:
        logging.debug(res)

    return res


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
 
    operand_datashape = blaze.dshape('10, 10, 10, float64')

    op0 = blaze.empty(operand_datashape)
    op1 = blaze.empty(operand_datashape)

    shape = (a.val for a in operand_datashape.shape)

    from operator import add
    t = time.time()
    for el in it_product(*[xrange(i) for i in shape]):
        val = random.uniform(-math.pi, math.pi)
        factor = math.sqrt(reduce(add, [j * 10**i for i, j in enumerate(reversed(el))]))

        op0[el] = math.sin(val) * factor
        op1[el] = math.cos(val) * factor

    logging.info("initialization took %f seconds", (time.time()-t))

    expr = op0*op0 + op1*op1

    eval_in_mem(expr, 0, dump_result=True)   
    eval_in_mem(expr, 0, dump_result=True)
    eval_in_mem(expr, iter_dims=1, dump_result=True)
    eval_in_mem(expr, iter_dims=1, chunk=3, dump_result=True)
    eval_in_mem(expr, iter_dims=1, chunk=4, dump_result=True)
    eval_in_mem(expr, iter_dims=1, chunk=5, dump_result=True)


    stor = blaze.Storage('blz://persisted.blz')
    t = time.time()
    result = blaze.eval(expr, storage=stor)
    logging.info("evaluation blz took %f seconds", time.time()-t)
    logging.debug(str(result))
    blaze.drop(stor)

    t = time.time()
    result = blaze.eval(expr)
    logging.info("evaluation hierarchical %f seconds", time.time()-t)
    logging.debug(str(result))

