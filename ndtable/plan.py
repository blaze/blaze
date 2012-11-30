"""
Execution plans.
"""
import string
import numpy as np
from mmap import PAGESIZE
from pprint import pprint

from collections import namedtuple
from ndtable.expr.graph import Literal, OP, APP, VAL
from ndtable.idx import Indexable
from ndtable.byteproto import CONTIGUOUS, READ
from ndtable.expr.paterm import AAppl, ATerm
from ndtable.expr.visitor import MroTransformer

L2SIZE = 2**17
L3SIZE = 2**20

L2   = 1
L3   = 2
PAGE = 3

map_kernels = {
    ('add', 'i', 'i'): ''
}

reduce_kernels = {
    ('sum', 'i', 'i'): ''
}

special_kernels = {
}

#------------------------------------------------------------------------
# Toy Kernels
#------------------------------------------------------------------------

def add_scalars(a,b,o):
    res = np.empty(1, o)
    np.add(a,b,res)
    return res

def add_incore(dd_a, ds_a, dd_b, ds_b, ds_o):
    res = np.empty(ds_o)
    np.add(ds_a.getbuffer(), ds_b.getbuffer(), res)
    return res

# chunks = [((0,1024), (0, 1024))], a range for each of the
# operands
def add_outcore(dd_a, ds_a, dd_b, ds_b, ds_o, chunks):
    res = np.empty(ds_o) # may be a scalar

    for ca_start, ca_end, cb_start, cb_end in chunks:
        np.add(
            ds_a.getchunk(ca_start, ca_end),
            ds_b.getchunk(cb_start, cb_end),
        res)

    return res

#------------------------------------------------------------------------
# Plans
#------------------------------------------------------------------------

class GenericInstruction(object):
    def __init__(self, op, ret=None, *args):
        self.op = op
        self.args = args

    def __repr__(self):
        return ' '.join([self.op,] + map(repr, self.args))

class Constant(object):
    def __init__(self, val):
        self.val = val

class BlazeVisitor(MroTransformer):
    def __init__(self):
        pass

    def App(self, graph):
        return self.visit(graph.children[0])

    def Op(self, graph):
        return AAppl(
            graph.__class__.__name__,
            self.visit(graph.children)
        )

    def Literal(self, graph):
        return ATerm(graph.val)

    def Indexable(self, graph):
        return ATerm(graph.data)


def generate(graph, variables, kernels):
    # The variables come in topologically sorted, so we just
    # have to preserve that order

    visitor = BlazeVisitor()
    return visitor.visit(graph[-1])
