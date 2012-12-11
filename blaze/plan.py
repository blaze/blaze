"""
Execute raw graph to ATerm after inference but before evaluation.
"""

import string
import numpy as np

from collections import namedtuple

#from blaze.expr.graph import Literal, OP, APP, VAL
#from blaze.table import Indexable
from blaze.datashape.coretypes import DataShape
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

def annotate_dshape(ds):
    """
    Convert a datashape instance into Aterm annotation

    >>> ds = dshape('2, 2, int32')
    >>> anno = dshape_anno(ds)
    dshape("2, 2, int32")
    >>> type(anno)
    <class 'AAppl'>
    """

    assert isinstance(ds, DataShape)
    return AAppl(ATerm('dshape'), [AString(str(ds))])

def annotation(graph, *metadata):
    # Metadata holds a reference to the graph node, not really
    # what we want but fine for now...
    metadata = (id(graph),) + metadata

    # was originally .datashape but this is a reserved attribute
    # so moved to a new simple_type() method that wraps around
    # promote()
    anno = annotate_dshape(graph.simple_type())
    annotation = AAnnotation(anno, metadata)
    return annotation

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------

class InstructionGen(MroVisitor):
    """ Map ATerm into linear instructions, unlike ATerm this
    does not preserve the information contained in the expression
    graph, information is discarded. """

    def AAppl(self):
        pass

class BlazeVisitor(MroVisitor):
    """ Map Blaze graph objects into ATerm """

    def __init__(self):
        self.operands = []

    def App(self, graph):
        return self.visit(graph.children[0])

    def Op(self, graph):
        opname = graph.__class__.__name__

        if graph.is_arithmetic:
            return AAppl(ATerm('Arithmetic'),
                         [opname] + self.visit(graph.children),
                         annotation=annotation(graph))
        else:
            return AAppl(opname, self.visit(graph.children),
                         annotation=annotation(graph))

    def Literal(self, graph):
        return ATerm(graph.val, annotation=annotation(graph))

    def Indexable(self, graph):
        self.operands.append(graph)
        return AAppl(ATerm('Array'), [], annotation=annotation(graph))

    def Slice(self, graph):
        # Slice(start, stop, step){id(graph), 'get'|'set'}
        array, start, stop, step = graph.operands

        if start:
            start = self.visit(start)
        if stop:
            stop = self.visit(stop)
        if step:
            step = self.visit(step)

        return AAppl(
            ATerm('Slice'),
            [self.visit(array),
             start or ATerm('None'),
             stop or ATerm('None'),
             step or ATerm('None')],
            annotation=annotation(graph, graph.op)
        )

    def IndexNode(self, graph):
        return AAppl(ATerm('Index'), self.visit(graph.operands),
                     annotation=annotation(graph, graph.op))

    def Assign(self, graph):
        return AAppl(ATerm('Assign'), self.visit(graph.operands),
                     annotation=annotation(graph))
