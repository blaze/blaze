"""
Holds the Blaze expression objects.

* App
* Op
* ArrayNode
* IntNode
* FloatNode
* StringNode
* Assign
* ExpressionNode
* Fun
* FunApp
* IndexNode

These form the function terms in the ATerm representation of the
graph.
"""

from numbers import Integral
from functools import partial
from collections import Iterable

from blaze import catalog
from blaze.expr import nodes
from blaze.eclass import eclass
from blaze.sources.canonical import PythonSource
from blaze.datashape import coretypes as T
from blaze.graph_kinds import graph_kind

#------------------------------------------------------------------------
# Kinds
#------------------------------------------------------------------------

OP  = graph_kind.OP
APP = graph_kind.APP
VAL = graph_kind.VAL
FUN = graph_kind.FUN

#------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------

_max_argument_recursion = 25
_max_argument_len       = 1000
_argument_sample        = 100

def set_max_argument_len(val):
    global _max_argument_len
    _max_argument_len = val

def set_max_argument_recursion(val):
    global _max_argument_recursion
    _max_argument_recursion = val

#------------------------------------------------------------------------
# Argument Munging
#------------------------------------------------------------------------

instanceof = lambda T: lambda X: isinstance(X, T)

def is_homogeneous(it):
    # Checks Python types for arguments, not to be confused with the
    # datashape types and the operator types!

    head = it[0]
    head_type = type(head)
    return head, all(type(a) == head_type for a in it)

def injest_iterable(args, depth=0, force_homog=False):
    # TODO: Should be 1 stack frame per each recursion so we
    # don't blow up Python trying to parse big structures
    assert isinstance(args, (list, tuple))

    if depth > _max_argument_recursion:
        raise RuntimeError(\
        "Maximum recursion depth reached while parsing arguments")

    # tuple, list, dictionary, any recursive combination of them
    if isinstance(args, Iterable):

        if len(args) == 0:
            return []

        if len(args) < _max_argument_len:
            sample = args[0:_argument_sample]

            # If the first 100 elements are type homogenous then
            # it's likely the rest of the iterable is.
            head, is_homog = is_homogeneous(sample)

            if force_homog and not is_homog:
                raise TypeError("Input is not homogenous.")

            # Homogenous Arguments
            # ====================

            if is_homog:
                if isinstance(head, int):
                    return [IntNode(a) for a in args]
                if isinstance(head, float):
                    return [FloatNode(a) for a in args]
                elif isinstance(head, basestring):
                    return [StringNode(a) for a in args]
                elif isinstance(head, ArrayNode):
                    return [a for a in args]
                else:
                    return args

            # Heterogenous Arguments
            # ======================

            # TODO: This will be really really slow, certainly
            # not something we'd want to put in a loop.
            # Optimize later!

            elif not is_homog:
                ret = []
                for a in args:
                    if isinstance(a, (list, tuple)):
                        sub = injest_iterable(a, depth+1)
                        ret.append(sub)
                    elif isinstance(a, ArrayNode):
                        ret.append(a)
                    elif isinstance(a, basestring):
                        ret.append(StringNode(a))
                    elif isinstance(a, int):
                        ret.append(IntNode(a))
                    elif isinstance(a, float):
                        ret.append(FloatNode(a))
                    elif isinstance(a, nodes.Node):
                        ret.append(a)
                    else:
                        raise TypeError("Unknown type")
                return ret

        else:
            raise RuntimeError("""
            Too many dynamic arguments to build expression
            graph. Consider alternative construction.""")

#------------------------------------------------------------------------
# Base Classes
#------------------------------------------------------------------------

class ExpressionNode(nodes.Node):
    """
    A abstract node which supports the full set of PyNumberMethods
    methods.
    """
    abstract = True

    def generate_opnode(self, arity, func_name, args=None, kwargs=None):

        iargs = injest_iterable(args)

        # TODO: better solution, resolve circular import
        from blaze.expr import ops

        # Lookup by capitalized name
        op = getattr(ops, func_name.capitalize())
        iop = op(func_name.capitalize(), iargs, kwargs)

        #op.__proto__

        return App(iop)

    # Python Intrinsics
    # -----------------
    for name in catalog.PyObject_Intrinsics:
        # Bound methods are actually just unary functions with
        # the first argument self implicit
        exec (
            "def __%(name)s__(self):\n"
            "    return self.generate_opnode(1, '%(name)s', [self])"
            "\n"
        ) % locals()
        del name

    # Unary
    # -----
    for name, _ in catalog.PyObject_UnaryOperators:
        exec (
            "def __%(name)s__(self):\n"
            "    return self.generate_opnode(1, '%(name)s', [self])"
            "\n"
        ) % locals()
        del name
        del _

    # Binary
    # ------
    for name, _ in catalog.PyObject_BinaryOperators:
        exec (
            "def __%(name)s__(self, ob):\n"
            "    return self.generate_opnode(2, '%(name)s', [self, ob])\n"
            "\n"
            "def __r%(name)s__(self, ob):\n"
            "    return self.generate_opnode(2, '%(name)s', [self, ob])\n"
            "\n"
        )  % locals()
        del name
        del _

    for name, _ in catalog.PyObject_BinaryOperators:
        exec (
            "def __i%(name)s__(self, ob):\n"
            "    return self.generate_opnode(2, '%(name)s', [self, ob])\n"
            "\n"
        )  % locals()
        del name
        del _

#------------------------------------------------------------------------
# Indexables
#------------------------------------------------------------------------

class ArrayNode(ExpressionNode):
    """
    A array structure with dimension and length.
    """
    kind = VAL

    # Read Operations
    # ===============

    for name in catalog.PyArray_ReadMethods:
        exec (
            "def %(name)s(self, *args, **kwargs):\n"
            "    args = (self,) + args\n"
            "    return self.generate_opnode(-1, '%(name)s', args, kwargs)"
            "\n"
        ) % locals()
        del name

    # Write Operations
    # ===============

    for name in catalog.PyArray_WriteMethods:
        exec (
            "def %(name)s(self, *args, **kwargs):\n"
            "    return self.generate_opnode(-1, '%(name)s', args, kwargs)\n"
            "\n"
        ) % locals()
        del name

    # Numpy-compatible shape/flag attributes
    # ======================================
    # These are evaluated in immediate mode, and do not return a deferred
    # graph node.  This implies that stream-generating functions (i.e. nodes
    # whose shape information requires a data evaluation) will actually
    # trigger an eval().

    @property
    def flags(self):
        pass

    @property
    def itemsize(self):
        pass

    @property
    def strides(self):
        pass

    @property
    def shape(self):
        pass

    @property
    def ndim(self):
        pass

    @property
    def size(self):
        pass

    # Numpy-compatible data attributes
    # ================================

    @property
    def imag(self):
        return Op('imag', self)

    @property
    def real(self):
        return Op('imag', self)

    @property
    def flat(self):
        """ Equivalent to .reshape(), which returns a graph node. """
        return Op('flat', self)

    @property
    def T(self):
        """ Equivalent to .transpose(), which returns a graph node. """
        return Op('transpose', self)

    # {get,set}item Operations
    # ===============

    def getitem(self, idx, context='get'):
        if isinstance(idx, slice):
            result = Slice(context, [self,
                                     intnode(idx.start),
                                     intnode(idx.stop),
                                     intnode(idx.step)])
        elif isinstance(idx, Integral):
            result = IndexNode(context, [self, idx])
        else:
            # TODO: detect other forms
            ndx = IndexNode(idx)
            result = Slice(context, [self, ndx])

        return result

    def __getitem__(self, idx):
        """ Slicing operations should return graph nodes, while individual
        element access should return bare scalars.
        """
        return self.getitem(idx)

    def __setitem__(self, idx, value):
        """
        This first creates a slice node in the expression graph to avoid
        an unnecessary RHS temporary. Evaluation is then immediate. We need
        this in order to avoid creating complicated dependence graphs:

            t = a + b
            a[...] = t
            result = t.eval()    # 't' refers to original 'a' and 'b'
                                 # we need to substitute the updated 'a'
                                 # for the original for this evaluation!
        """
        if not isinstance(value, nodes.Node):
            raise NotImplementedError("setitem with non-blaze rhs")
        result = self.getitem(idx, context='set')
        result = Assign('assign', [result, value])
        result.eval()

    # Other non-graph methods
    # ========================

    def tofile(self, *args, **kw):
        pass

    def tolist(self, *args, **kw):
        pass

    def tostring(self, *args, **kw):
        pass

def intnode(value_or_graph):
    if value_or_graph is None:
        return None
    elif isinstance(value_or_graph, nodes.Node):
        return value_or_graph
    else:
        return IntNode(value_or_graph)

#------------------------------------------------------------------------
# Application
#------------------------------------------------------------------------

class App(ExpressionNode):
    """
    Operator application.
    """
    kind = APP

    def __init__(self, operator):
        self.operator = operator
        self.children = [operator]

    @property
    def name(self):
        return 'App'

    def __repr__(self):
        return '<App(%s, %s)>' % (self.operator.name, [])

#------------------------------------------------------------------------
# Op
#------------------------------------------------------------------------

class Op(ExpressionNode):
    """
    A typed operator taking a set of typed operands.
    """
    kind = OP

    def __init__(self, op, operands, kwargs=None):
        self.op = op
        self.children = operands
        self.operands = operands
        self.kwargs = kwargs

        # TODO: type inference on the aterm graph
        from blaze import stopgap
        self.datashape = stopgap.compute_datashape(self, operands, kwargs)

    @property
    def name(self):
        return str(self.op)

    def __repr__(self):
        return '<Op(%s)>' % (self.name,)

#------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------

class Fun(ExpressionNode):
    """
    A generic function application. Normally constructed by
    @lift'ing a function into the Blaze runtime.
    """
    kind = FUN

    # nargs, fn, fname are spliced in at construction

    def __init__(self, arguments):
        if len(arguments) != self.nargs:
            raise TypeError('%s exepected at most %i args' % (self.fname, self.nargs))
        self.children = arguments

    @property
    def name(self):
        return str(self.fname)

#------------------------------------------------------------------------
# Values
#------------------------------------------------------------------------

class Literal(ExpressionNode):
    kind = VAL

    def __init__(self, val):
        assert isinstance(val, self.vtype)
        self.val = val
        self.children = []

    @property
    def name(self):
        return str(self.val)

#------------------------------------------------------------------------
# Strings
#------------------------------------------------------------------------

class StringNode(Literal):
    kind      = VAL
    vtype     = str
    datashape = T.string
    eclass    = eclass.manifest
    datashape = T.string

    @property
    def data(self):
        return PythonSource(self.val, type=str)

#------------------------------------------------------------------------
# Scalars
#------------------------------------------------------------------------

class IntNode(Literal):
    vtype     = int
    datashape = T.int_
    kind      = VAL
    eclass    = eclass.manifest
    datashape = T.int_

    @property
    def data(self):
        return PythonSource(self.val, type=int)

class FloatNode(Literal):
    kind      = VAL
    vtype     = float
    datashape = T.float_
    eclass    = eclass.manifest
    datashape = T.float_

    @property
    def data(self):
        return PythonSource(self.val, type=float)

#------------------------------------------------------------------------
# Slices and Indexes
#------------------------------------------------------------------------

class IndexNode(Op):
    kind  = OP
    arity  = 2 # <INDEXABLE>, <INDEXER>
    vtype = tuple

    @property
    def name(self):
        return 'Index%s' % str(self.val)

class Slice(Op):
    kind   = OP
    arity  = 4 # <INDEXABLE>, start, stop, step

class Assign(Op):
    kind   = OP
    arity  = 2

#------------------------------------------------------------------------
# Control Flow
#------------------------------------------------------------------------

class IfElse(Op):
    kind   = OP
    arity  = 3
