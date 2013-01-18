"""
Execute raw graph to ATerm after inference but before evaluation.
"""

from pprint import pformat

from blaze.rts.funcs import lookup
from blaze.expr.visitor import MroVisitor
from blaze.datashape import DataShape
from blaze.aterm import aappl, aterm, astr, aint, areal

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
    return aappl(aterm('dshape'), [astr(str(ds))])

#------------------------------------------------------------------------
# Plan Primitives
#------------------------------------------------------------------------

class Plan(object):
    def __init__(self, instructions):
        self.instructions = instructions

    def __repr__(self):
        return pformat(self.instructions)


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

#------------------------------------------------------------------------
# Instruction Generation
#------------------------------------------------------------------------

# TODO: plan file should be serializable and in the context of a
# symbol table uniquely defines the computation

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
            %1 : Array(...)
            %2 : Array(...)
            %3 : <alloca>
            %4 : <alloca>
        }

    """

    def __init__(self):
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
        label = term.spine.term

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
        self._vartable[term] = Constant(term.val)
        return

    def AReal(self, term):
        self._vartable[term] = Constant(term.val)
        return

    def ATerm(self, term):
        return

#------------------------------------------------------------------------
# Internal Blaze Graph -> ATerm
#------------------------------------------------------------------------

# TODO: Visit on KIND instead of class!

class BlazeVisitor(MroVisitor):
    """ Map Blaze graph objects into ATerm """

    def __init__(self):
        self.operands = []

    def App(self, node):
        return self.visit(node.operator)

    def Fun(self, node):
        return self.visit(node.children)

    def Op(self, node):
        opname = node.__class__.__name__
        return aappl(aterm(opname), self.visit(node.children))

    def Literal(self, node):
        if node.vtype == int:
            return aint(node.val)
        if node.vtype == float:
            return areal(node.val)
        else:
            return aterm(node.val)

    def Indexable(self, node):
        self.operands.append(node)
        return aappl(aterm('Array'), [])

    def Slice(self, node):
        # Slice(start, stop, step){id(node), 'get'|'set'}
        array, start, stop, step = node.operands

        if start:
            start = self.visit(start)
        if stop:
            stop = self.visit(stop)
        if step:
            step = self.visit(step)

        return aappl(
            aterm('Slice'),
            [self.visit(array),
             start or aterm('None'),
             stop  or aterm('None'),
             step  or aterm('None')],
        )

    def IndexNode(self, node):
        return aappl(aterm('Index'), self.visit(node.operands))

    def Assign(self, node):
        return aappl(aterm('Assign'), self.visit(node.operands))
