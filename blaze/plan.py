"""
Execute raw graph to ATerm after inference but before evaluation.
"""

import string
import numpy as np

from collections import namedtuple
from blaze.expr.graph import Literal, OP, APP, VAL
from blaze.idx import Indexable
from blaze.byteproto import CONTIGUOUS, READ

from blaze.expr.paterm import AAppl, ATerm, AAnnotation
from blaze.expr.visitor import MroVisitor

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

class BlazeVisitor(MroVisitor):
    def __init__(self):
        pass

    def App(self, graph):
        return self.visit(graph.children[0])

    def Op(self, graph):
        opname = graph.__class__.__name__

        if graph.is_arithmetic:
            return AAppl(ATerm('Arithmetic'),
                         [opname] +  self.visit(graph.children))
        else:
            return AAppl(opname, self.visit(graph.children))

    def Literal(self, graph):
        return ATerm(graph.val)

    def Indexable(self, graph):
        return AAnnotation(
            AAppl(ATerm('Array'), [ATerm(id(graph.data))]),
            ty=ATerm(repr(graph.datashape))
        )


def generate(graph, variables):
    # The variables come in topologically sorted, so we just
    # have to preserve that order

    visitor = BlazeVisitor()
    return visitor.visit(graph[-1])
