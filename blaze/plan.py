"""
Execute raw graph to ATerm after inference but before evaluation.
"""

import itertools
from pprint import pformat

from blaze.funcs import lookup
from blaze.visitor import MroVisitor
from blaze.datashape import DataShape, dshape
from blaze.aterm import aappl, aterm, astr, aint, areal, match

#------------------------------------------------------------------------
# Plans
#------------------------------------------------------------------------

def annotate_dshape(ds):
    """
    Convert a datashape instance into Aterm annotation

    >>> ds = dshape('2, 2, int32')
    >>> anno = annotate_dshape(ds)
    >>> anno
    dshape("2, 2, int32")
    >>> type(anno)
    <class 'blaze.aterm.terms.AAppl'>
    """

    assert isinstance(ds, DataShape)
    return aappl(aterm('dshape'), [astr(str(ds))])

def get_datashape(term):
    """
    >>> ds = dshape('2, 2, int32')
    >>> ds2 = get_datashape(annotate_dshape(ds))
    >>> type(ds2)
    <class 'blaze.datashape.coretypes.DataShape'>
    >>> ds2
    dshape("2, 2, int32")
    """
    # result = match("dshape(<value>)", term)
    # assert result, result
    # dshape_string = result['value'].val
    dshape_string = term.args[0].val
    return dshape(dshape_string)

def annotate(node, metadata):
    if node.annotation is None:
        node.annotation = metadata
    else:
        node.annotation.extend(metadata)

    return node
    # return aappl(aterm('Annotation'), [node] + metadata)

#------------------------------------------------------------------------
# Plan Primitives
#------------------------------------------------------------------------

# %2 = multiply %0 %1
# %5 = multiply %3 %4
# %6 = add %2 %5
# %9 = multiply %7 %8

class Plan(object):
    def __init__(self, vartable, instructions):
        self.vartable = vartable
        self.instructions = instructions

    def tofile(self, fname, local=False):
        with open(fname, 'w+') as fd:
            for var in self.vars:
                fd.write(var.geturi(local))
            fd.write('\r\n')
            for ins in self.instructions:
                fd.write(str(ins))
            fd.write('\n')

    def fromfile(self):
        raise NotImplementedError

    def __repr__(self):
        return pformat(self.instructions)

class Constant(object):
    def __init__(self, n):
        self.n = n

    def __repr__(self):
        return 'Const(%s)' % self.n

class Var(object):
    def __init__(self, key):
        self.key = key

    def __repr__(self):
        return self.key

class Instruction(object):
    def __init__(self, fn, datashape, args=None, lhs=None, fillvalue=None):
        # %lhs = fn{props}(arguments)

        self.fn = fn
        self.datashape = datashape
        self.args = args or []
        self.lhs = lhs
        self.fillvalue = fillvalue

    def execute(self, operands, lhs=None):
        return self.fn(operands, lhs)

    def __repr__(self):
        # with output types
        if self.lhs:
            return self.lhs + ' = ' + \
            ' '.join([self.fn.name] + map(str, self.args))
        # purely side effectful
        else:
            return ' '.join([self.fn] + map(str, self.args))

#------------------------------------------------------------------------
# Instruction Generation
#------------------------------------------------------------------------

# TODO: plan file should be serializable and in the context of a
# symbol table uniquely defines the computation

class InstructionGen(MroVisitor):
    """ Map ATerm into linear instructions.

    Does pre-order traversal of the expression graph and maintains a
    stack as the nodes are visited, the instructions for the innermost
    term are top on the stack. The temporaries are mapped through the
    vartable.

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

    def __init__(self, executors):
        self.executors = executors

        self._vartable = {}
        self._instructions = []
        self._vars = ('%' + str(n) for n in itertools.count(0))

    def tmp(self):
        return next(self._vars)

    def push(self, ins):
        self._instructions.append(ins)

    @property
    def tos(self):
        return self._instructions[-1]

    @property
    def plan(self):
        return self._instructions

    @property
    def vars(self):
        return self._vartable

    @property
    def symbols(self):
        return dict((name, term) for term, name in self._vartable.iteritems())

    def var(self, term):
        key = self.tmp()
        self._vartable[term] = key
        return key

    def _Op(self, term):
        spine = term.spine
        args  = term.args
        dshape_term = term.annotation[0]

        # visit the innermost arguments, push those arguments on
        # the stack first
        map(self.visit, term.args)

        fn, cost = lookup(term)

        fargs = [self._vartable[a] for a in args]

        # push the temporary for the result in the vartable
        key = self.var(term)

        # build the instruction & push it on the stack
        inst = Instruction(fn, get_datashape(dshape_term), fargs, lhs=key)
        self.push(inst)

    def _Array(self, term):
        key = self.var(term)
        # return Var(key)

    def _Assign(self, term):
        pass

    def _Slice(self, term):
        pass

    def _Executor(self, term):
        executor_id, backend, has_lhs, fillvalue = term.annotation.meta
        has_lhs = has_lhs.label
        fillvalue = fillvalue.label
        executor = self.executors[executor_id.label]

        self.visit(term.args)

        fargs = [self._vartable[a] for a in term.args]

        if has_lhs:
            fargs, lhs = fargs[:-1], fargs[-1]
        else:
            lhs = None

        # build the instruction & push it on the stack
        inst = Instruction(executor, get_datashape(term), fargs, lhs=lhs,
                           fillvalue=fillvalue)
        self.push(inst)

    def AAppl(self, term):
        label = term.spine.term

        if label == 'Array':
            return self._Array(term)
        elif label == 'Slice':
            return self._Slice(term)
        elif label == 'Assign':
            return self._Assign(term)
        elif label == 'Executor':
            return self._Executor(term)
        else:
            return self._Op(term)

    def AInt(self, term):
        const = Constant(term.val)
        self._vartable[term] = const
        return const

    def AReal(self, term):
        const = Constant(term.val)
        self._vartable[term] = const
        return const

    def ATerm(self, term):
        raise NotImplementedError

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
        result_node = aappl(aterm(opname), self.visit(node.children))
        return annotate(result_node, [annotate_dshape(node.datashape)])

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

if __name__ == '__main__':
    import doctest
    doctest.testmod()