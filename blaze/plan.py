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
# ATerm -> Instructions
#------------------------------------------------------------------------

class Instruction(object):
    def __init__(self, fn, lhs=None, *args):
        """ %lhs = fn{props}(arguments) """

        self.fn = fn
        self.args = args
        self.lhs = lhs

    def __repr__(self):
        # with output types
        if self.ret:
            return self.ret + ' = ' + ' '.join((self.fn,) + self.args)
        # purely side effectful
        else:
            return ' '.join([self.fn,] + map(repr, self.args))

class InstructionGen(MroVisitor):
    """ Map ATerm into linear instructions, unlike ATerm this
    does not preserve the information contained in the expression
    graph, information is discarded. """

    def __init__(self, have_numbapro):
        self.numbapro = have_numbapro

    def AAppl(self, term):
        from blaze.rts.ffi import lookup

        # All the function signatures are of the form
        #
        #     Add(a,b)
        #
        # But the aterm expression for LLVM is expected to be
        #
        #     Arithmetic(Add, ...)
        #
        # so we do this ugly hack to get the signature back to
        # standard form

        # -- hack --
        op   = term.args[0]
        args = term.args[1:]
        normal_term = AAppl(ATerm(op), args)
        # --

        # ugly hack because construction of AAppl is inconsistent
        # and code is written against the inconsistency :(
        #
        # good:     AAppl(ATerm('x'), ...)
        # not good: AAppl('x')

        if isinstance(op, basestring):
            spine = op
        elif isinstance(op, AAppl):
            spine = op.spine.label
        else:
            raise NotImplementedError

        if spine in {'Slice', 'Assign'}:
            return []

        if self.numbapro:
            pass
            # ==================================================
            # TODO: right here is where we would call the
            # ExecutionPipeline and build a numba ufunc kernel
            # if we have numbapro. We would pass in the original
            # ``term`` object which is still of the expected form:
            #
            #      Arithmetic(Add, ...)
            # ==================================================

        # otherwise, go find us implementation for how to execute
        # Returns either a ForeignF ( reference to a external C
        # library ) or a PythonF, a Python callable. These can be
        # anything, numpy ufuncs, numexpr, pandas, cmath whatever
        x = lookup(normal_term)

        # wrap it upin a instruction
        return []

#------------------------------------------------------------------------
# Graph -> ATerm
#------------------------------------------------------------------------

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
