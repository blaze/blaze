"""
Execute raw graph to ATerm after inference but before evaluation.
"""

import string
import numpy as np

from collections import namedtuple

from blaze.datashape.coretypes import DataShape
from blaze.byteproto import CONTIGUOUS, READ

from blaze.expr.paterm import AAppl, ATerm, AAnnotation, AString, AInt, AFloat
from blaze.expr.visitor import MroVisitor

#------------------------------------------------------------------------
# Plans
#------------------------------------------------------------------------

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

class Constant(object):
    def __init__(self, n):
        self.n = n
    def __repr__(self):
        return 'const(%s)' % self.n

class Var(object):
    def __init__(self, key):
        self.key = key
    def __repr__(self):
        return self.key

class Instruction(object):
    def __init__(self, fn, args=None, lhs=None):
        """ %lhs = fn{props}(arguments) """

        self.fn = fn
        self.lhs = lhs
        self.args = args or []

    def __repr__(self):
        # with output types
        if self.lhs:
            return self.lhs + ' = ' + \
            ' '.join([self.fn,] + map(repr, self.args))
        # purely side effectful
        else:
            return ' '.join([self.fn,] + map(repr, self.args))

# TODO: plan file should be serializable and in the context of a
# symbol table uniquely defines the computation

class Plan(object):
    def __init__(self, instructions):
        self.instructions = instructions

    def __repr__(self):
        return pformat(self.instructions)

# TODO: naive constant folding

class InstructionGen(MroVisitor):
    """ Map ATerm into linear instructions, unlike ATerm this
    does not preserve the information contained in the expression
    graph, information is discarded.

    Maintains a stack as the nodes are visited, the instructions
    for the innermost term are top on the stack. The temporaries
    are mapped through the vartable.

    ::

        a + b * c

    ::

        instructions = [
            %3 = <ufunc 'multiply'> %1 %2,
            %4 = <ufunc 'add'> %0 %3
        ]

        vartable = {
            Array(){dshape("2, 2, int32"),54490464}   : '%0',
            Array(){dshape("2, 2, float32"),54490176} : '%1',
            Array(){dshape("2, 2, int32"),54491184}   : '%2',
            ...
        }

    """

    def __init__(self, have_numbapro):
        self.numbapro = have_numbapro

        self.n = 0
        self._vartable = {}
        self._instructions = []

    def plan(self):
        return self._instructions

    @property
    def vars(self):
        return self._vartable

    def var(self, term):
        key = ('%' + str(self.n))
        self._vartable[term] = key
        self.n += 1
        return key

    def AAppl(self, term):
        label = term.spine.label

        if label == 'Array':
            return self._Array(term)
        elif label == 'Slice':
            return self._Slice(term)
        elif label == 'Assign':
            return self._Assign(term)
        else:
            return self._Op(term)

    def _Op(self, term):
        spine = term.spine
        args  = term.args

        # otherwise, go find us implementation for how to execute
        # Returns either a ExternalF ( reference to a external C
        # library ) or a PythonF, a Python callable. These can be
        # anything, numpy ufuncs, numexpr, pandas, cmath whatever
        from blaze.rts.funcs import lookup

        # visit the innermost arguments, push those arguments on
        # the instruction list first
        self.visit(term.args)

        fn, cost = lookup(term)
        fargs = [self._vartable[a] for a in args]

        # push the temporary for the result in the vartable
        key = self.var(term)

        # build the instruction & push it on the stack
        inst = Instruction(str(fn.fn), fargs, lhs=key)
        self._instructions.append(inst)

    def _Array(self, term):
        key = self.var(term)
        return Var(key)

    def _Assign(self, term):
        pass

    def _Slice(self, term):
        pass

    def AInt(self, term):
        self._vartable[term] = Constant(term.n)
        return

    def AFloat(self, term):
        self._vartable[term] = Constant(term.n)
        return

    def ATerm(self, term):
        return

#------------------------------------------------------------------------
# Graph -> ATerm
#------------------------------------------------------------------------

class BlazeVisitor(MroVisitor):
    """ Map Blaze graph objects into ATerm """

    def __init__(self):
        self.operands = []

    def App(self, graph):
        return self.visit(graph.operator)

    def Fun(self, graph):
        return self.visit(graph.children)

    def Op(self, graph):
        opname = graph.__class__.__name__
        return AAppl(ATerm(opname), self.visit(graph.children))

    def Literal(self, graph):
        if graph.vtype == int:
            return AInt(graph.val)
        if graph.vtype == float:
            return AFloat(graph.val)
        else:
            return ATerm(graph.val)

    def Indexable(self, graph):
        self.operands.append(graph)
        return AAppl(ATerm('Array'), [])

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
             stop  or ATerm('None'),
             step  or ATerm('None')],
        )

    def IndexNode(self, graph):
        return AAppl(ATerm('Index'), self.visit(graph.operands))

    def Assign(self, graph):
        return AAppl(ATerm('Assign'), self.visit(graph.operands))
