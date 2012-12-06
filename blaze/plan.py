"""
Execute raw graph to ATerm after inference but before evaluation.
"""

import string
import numpy as np

from collections import namedtuple

from blaze.expr.graph import Literal, OP, APP, VAL
from blaze.table import Indexable
from blaze.byteproto import CONTIGUOUS, READ

from blaze.expr.paterm import AAppl, ATerm, AAnnotation, AString
from blaze.expr.visitor import MroVisitor

#------------------------------------------------------------------------
# Plans
#------------------------------------------------------------------------

# Annotate with simple_type() which will pull the simple type of
# the App, Op, or Literal node before we even hit eval(). This is
# a stop gap measure because right we're protoyping this without
# having the types annotated on the graph that we're using for
# Numba code generation. It's still probably a good idea to have
# this knowledge available if we have it though!

def annotation(graph, *metadata):
    metadata = (id(graph),) + metadata
    # was originally .datashape but this is a reserved attribute
    # so moved to a new simple_type() method that wraps around
    # promote()
    annotation = AAnnotation(AString(str(graph.simple_type())), metadata)
    return annotation

class BlazeVisitor(MroVisitor):
    def __init__(self):
        self.operands = []

    def App(self, graph):
        return self.visit(graph.children[0])

    def Op(self, graph):
        opname = graph.__class__.__name__

        if graph.is_arithmetic:
            return AAppl(ATerm('Arithmetic'),
                         [opname] +  self.visit(graph.children),
                         annotation=annotation(graph))
        else:
            return AAppl(opname, self.visit(graph.children),
                         annotation=annotation(graph))

    def Literal(self, graph):
        return ATerm(graph.val, annotation=annotation(graph))

    def Indexable(self, graph):
        self.operands.append(graph)
        return AAppl(ATerm('Array'), [], annotation=annotation(graph))


def generate(graph, variables):
    # The variables come in topologically sorted, so we just
    # have to preserve that order

    visitor = BlazeVisitor()
    result = visitor.visit(graph[-1])
    return visitor.operands, result
