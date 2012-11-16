"""
Execution plans, ala SQL.
"""

# TODO: for bootstrapping
import math
import numpy as np
import carray as ca
from ndtable.expr.graph import VAL, OP, APP

from collections import Sequence, OrderedDict, namedtuple

READ  = 1
WRITE = 2

def traverse(x):
    if isinstance(x, dict):
        for a in x.k:
            yield a
    else:
        yield x

#------------------------------------------------------------------------
# Execution Plans
#------------------------------------------------------------------------

class Operation(object):

    def __init__(self, kernel, descriptors, seq=False, csections=None):
        self.ordered = False

        self.sequential = seq
        self.parallel = not self.sequential

        # later...
        #if self.parallel:
            #self.csections = []

#------------------------------------------------------------------------
# Ops
#------------------------------------------------------------------------

def coerce_descriptor(kernel, operands):
    """
    Coerce a descriptor into what the kernel desires. Produces a
    sequence of operations that load that load the bytes into a
    collection of temporaries if the underlying medium doesn't
    support the optimal operation ( i.e casting a Stream into a
    Contigious ).

        s : Stream(1,2,3) :: Contigious =

            [
                tmp = allocate(3, int32),
                tmp <- read(s),
                tmp <- read(s),
                tmp <- read(s)
            ]

    If the descriptor is optimal this does nothing.
    """
    # XXX
    return operands

def generate(ops, vars, kernels, hints):
    """ Generate the execution plan for retrieval of data and
    execution of kernels """
    _vars = {}

    for var in vars:
        if var.kind == VAL:
            descriptor = var.data
            _vars[var] = descriptor

    for op in ops:
        kernel = kernels[op]
        lvars = coerce_descriptor(kernel, *ops)
        # XXX -- kernel specialization here --

        yield Operation(kernel, lvars)

def showplan(plan, indent=0):
    """
    Pretty print the execution plan with ordering.

    S add
    S mul
    S abs

    """
    for operation in plan:
        flag = 'P' if (operation.parallel) else 'S'
        return (' '*indent) + flag + ( + showplan(operation))
