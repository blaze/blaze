"""
Core of the deferred expression engine.
"""

from functools import wraps
from numbers import Integral
from collections import Iterable

from ndtable.expr.nodes import Node
from ndtable.datashape.unification import unify
from ndtable.datashape.coretypes import int32, float32, top, Any

# conditional import of Numpy; if it doesn't exist, then set up dummy objects
# for the things we use
try:
    import numpy as np
except ImportError:
    np = {"integer": Integral}

#------------------------------------------------------------------------
# Globals
#------------------------------------------------------------------------

_max_argument_recursion = 25
_max_argument_len       = 1000
_argument_sample        = 100
_perform_typecheck      = True

def set_max_argument_len(val):
    global _max_argument_len
    _max_argument_len = val

def set_max_argument_recursion(val):
    global _max_argument_recursion
    _max_argument_recursion = val

#------------------------------------------------------------------------
# Method Maps
#------------------------------------------------------------------------

PyObject_BinaryOperators = [
    ('or','|'),  ('and','&'), ('xor','^'), ('lshift','<<'), ('rshift','>>'),
    ('add','+'), ('sub','-'), ('mul','*'), ('div','/'), ('mod','%'),
    ('truediv','/'), ('floordiv','//'), ('lt','<'), ('gt','>'), ('le','<='),
    ('ge','>='), ('eq','=='), ('ne','!=')
]

PyObject_UnaryOperators = [
    ('neg','-'), ('pos','+'), ('invert','~')
]

PyObject_Intrinsics = [
    'str', 'hash', 'abs', 'complex', 'int', 'long', 'float', 'iter',
    'oct', 'hex'
]

PyArray_Intrinsics = [
    "dtype", "size"
]

PyArray_WriteMethods = [
    'fill', 'itemset', 'put', 'setflags', 'setfield'
]

PyArray_ReadMethods = [
    'all', 'any', 'argmax', 'argmin', 'argsort', 'astype', 'base', 'byteswap',
    'choose', 'clip', 'compress', 'conj', 'conjugate', 'copy', 'ctypes',
    'cumprod', 'cumsum', 'data', 'diagonal', 'dot', 'dtype', 'dump', 'dumps',
    'flatten', 'getfield', 'item', 'max', 'mean', 'min', 'nbytes',
    'newbyteorder', 'nonzero', 'prod', 'ptp', 'ravel', 'repeat', 'reshape',
    'resize', 'round', 'searchsorted', 'setasflat', 'sort', 'squeeze', 'std',
    'sum', 'swapaxes', 'take', 'trace', 'transpose', 'var', 'view'
]

def lift_magic(f):
    @wraps(f)
    def fn(*args):
        iargs = injest_iterable(args)
        return f(*iargs)
    return fn

#------------------------------------------------------------------------
# Deconstructors
#------------------------------------------------------------------------

class UnknownExpression(Exception):
    def __init__(self, obj):
        self.obj = obj
    def __str__(self):
        return 'Unknown object in expression: %r' % (self.obj,)

def typeof(obj):
    # TODO: resolve circular dependencies
    from ndtable.table import NDTable

    if type(obj) is IntNode:
        return int32
    elif type(obj) is DoubleNode:
        return float32
    elif type(obj) is NDTable:
        return top
    elif type(obj) is Any:
        return top
    elif type(obj) is App:
        return obj.otype
    else:
        raise UnknownExpression(obj)

def dynamic(cls):
    universal = set([top])
    return all(arg == universal for arg in cls.dom)

#------------------------------------------------------------------------
# Graph Construction
#------------------------------------------------------------------------

def is_homogeneous(it):
    # type() comparisions are cheap pointer arithmetic on
    # PyObject->tp_type, isinstance() calls are expensive since
    # they have travese the whole mro hierarchy

    head = it[0]
    head_type = type(head)
    return head, all(type(a) == head_type for a in it)

def injest_iterable(args, depth=0):
    # TODO: Should be 1 stack frame per each recursion so we
    # don't blow up Python trying to parse big structures

    if depth > _max_argument_recursion:
        raise RuntimeError(\
        "Maximum recursion depth reached while parsing arguments")

    # tuple, list, dictionary, any recursive combination of them
    if isinstance(args, Iterable):

        # TODO: resolve circular dependencies
        from ndtable.table import NDTable

        if len(args) == 0:
            return []

        if len(args) < _max_argument_len:
            sample = args[0:_argument_sample]

            # If the first 100 elements are type homogenous then
            # it's likely the rest of the iterable is.
            head, is_homog = is_homogeneous(sample)
            is_hetero = not is_homog


            # Homogenous Arguments
            # ====================

            if is_homog:
                if isinstance(head, int):
                    return [IntNode(a) for a in args]
                if isinstance(head, float):
                    return [DoubleNode(a) for a in args]
                elif isinstance(head, basestring):
                    return [StringNode(a) for a in args]
                elif isinstance(head, NDTable):
                    return [a for a in args]
                else:
                    return args

            # Heterogenous Arguments
            # ======================

            # TODO: This will be really really slow, certainly
            # not something we'd want to put in a loop.
            # Optimize later!

            elif is_hetero:
                ret = []
                for a in args:
                    if isinstance(a, (list, tuple)):
                        sub = injest_iterable(a, depth+1)
                        ret.append(sub)
                    elif isinstance(a, NDTable):
                        ret.append(a)
                    elif isinstance(a, Node):
                        ret.append(a)
                    elif isinstance(a, basestring):
                        ret.append(StringNode(a))
                    elif isinstance(a, int):
                        ret.append(IntNode(a))
                    elif isinstance(a, float):
                        ret.append(DoubleNode(a))
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

class ExpressionNode(Node):
    """
    A abstract node which supports the full set of PyNumberMethods
    methods.
    """

    def eval(self):
        pass

    def generate_node(self, arity, fname, args=None, kwargs=None):

        # TODO: factor out circular imports
        from ndtable.expr.ops import functions

        # TODO: also kwargs when we support such things
        iargs = injest_iterable(args)

        #op = functions[fname]
        op = functions.get(fname, Op)

        if arity == 1:
            assert len(iargs) == arity
            return op(fname, iargs)

        if arity == 2:
            assert len(iargs) == arity
            return op(fname, iargs)

        elif arity == -1:
            return op(fname, iargs, kwargs)

    # Python Intrinsics
    # -----------------
    for name in PyObject_Intrinsics:
        # Bound methods are actually just unary functions with
        # the first argument self implicit
        exec (
            "def __%(name)s__(self):\n"
            "    return self.generate_node(1, '%(name)s', [self])"
            "\n"
        ) % locals()

    # Unary
    # -----
    for name, op in PyObject_UnaryOperators:
        exec (
            "def __%(name)s__(self):\n"
            "    return self.generate_node(1, '%(name)s', [self])"
            "\n"
        ) % locals()

    # Binary
    # ------
    for name, op in PyObject_BinaryOperators:
        exec (
            "def __%(name)s__(self, ob):\n"
            "    return self.generate_node(2, '%(name)s', [self, ob])\n"
            "\n"
            "def __r%(name)s__(self, ob):\n"
            "    return self.generate_node(2, '%(name)s', [self, ob])\n"
            "\n"
        )  % locals()

    for name, op in PyObject_BinaryOperators:
        exec (
            "def __i%(name)s__(self, ob):\n"
            "    return self.generate_node(2, '%(name)s', [self, ob])\n"
            "\n"
        )  % locals()

#------------------------------------------------------------------------
# Indexables
#------------------------------------------------------------------------

class ArrayNode(ExpressionNode):
    """
    A array structure with dimension and length.
    """

    # Read Operations
    # ===============

    for name in PyArray_ReadMethods:
        exec (
            "def %(name)s(self, *args, **kwargs):\n"
            "    args = (self,) + args\n"
            "    return self.generate_node(-1, '%(name)s', args, kwargs)"
            "\n"
        ) % locals()

    # Write Operations
    # ===============

    for name in PyArray_WriteMethods:
        exec (
            "def %(name)s(self, *args, **kwargs):\n"
            "    return self.generate_node(-1, '%(name)s', args, kwargs)\n"
            "\n"
        ) % locals()

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

    @property
    def dtype(self):
        pass

    def __len__(self):
        # TODO: needs to query datashape
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

    # Read Operations
    # ===============

    def __getitem__(self, idx):
        """ Slicing operations should return graph nodes, while individual
        element access should return bare scalars.
        """
        if isinstance(idx, Integral) or isinstance(idx, np.integer):
            ndx = IndexNode((idx,))
            return Slice('getitem', [self, ndx])
        else:
            ndx = IndexNode(idx)
            return Slice('getitem', [self, ndx])

    def __getslice__(self, start, stop):
        """
        """
        ndx = IndexNode((start, stop))
        return Slice('getslice', [self, ndx])

    # Other non-graph methods
    # ========================

    def tofile(self, *args, **kw):
        pass

    def tolist(self, *args, **kw):
        pass

    def tostring(self, *args, **kw):
        pass

#------------------------------------------------------------------------
# Transformational
#------------------------------------------------------------------------

class App(ExpressionNode):
    """
    The application of an operand producing concrete values.

    For example:

        In[0]: a = 2 + 3

    The resulting value of ``a`` is App( Op(+), 2, 3) the
    signature of the application is with output type int32.

    ::

                       +----------------+
                       |       +----+   |
                       |    / -|    |   |
        ival :: op.cod |---+  -| Op |---| -> oval :: op.dom
                       |    \ -|    |   |
                       |       +-----   |
                       +----------------+

    """
    __slots__ = ['itype','otype']

    def __init__(self, op, inputs, outputs):
        self.op = op
        self.children = [op]

    def nin(self):
        return len(self.itype)

    def nout(self):
        return len(self.otype)

class Op(ExpressionNode):
    """
    A typed operator taking a set of typed operands. Optionally
    rejects operands which are not well-typed.

              a -> b -> c -> d

                    +---+
        op1 :: a -> |   |
        op2 :: b -> | f | -> * :: d
        op3 :: c -> |   |
                    +---+
    """
    __slots__ = ['children', 'op', 'cod']

    @property
    def opaque(self):
        """
        We don't know anything about the operator, no types, no argument
        signature ... we just throw things into and things pop out or it
        blows up.
        """
        return not hasattr(self, 'signature')

    def __init__(self, op, operands):
        self.op = op
        self.children = operands

        # Make sure the graph makes sense given the signature of
        # the function. Does naive type checking and inference.
        if _perform_typecheck and not self.opaque:
            env = self.typecheck(operands)
            self.cod = self.infer_codomain(env)
        else:
            # Otherwise it's the universal supertype, the operator could
            # return anything. Usefull for when we don't know much about
            # the operand a priori
            self.cod = top

    def infer_codomain(self, env):
        tokens = [
            tok.strip()
            for tok in
            self.signature.split('->')
        ]

        cod = tokens[-1]
        return env[cod]

    def typecheck(self, operands):
        # TODO: commutative operands!
        # We can use itertools.permutations to permute domains

        if not hasattr(self, 'signature'):
            return True

        # A dynamically typed function, just drop out before
        # hitting the expensive type checker since we know the
        # codomain is dynamic as well.
        #if dynamic(self):
            #return True

        tokens = [
            tok.strip()
            for tok in
            self.signature.split('->')
        ]

        dom = tokens[0:-1]
        cod = tokens[-1]

        # Rigid variables are those that appear in multiple times
        # in the signature
        #      a -> b -> a -> c
        #      Rigid a
        #      Free  b,c
        rigid = [tokens.count(token)  > 1 for token in dom]
        free  = [tokens.count(token) == 1 for token in dom]

        assert len(dom) == self.arity
        env = {}

        dom_vars = dom

        # Naive Type Checker
        # ==================

        for i, (var, types, operand) in enumerate(zip(dom_vars, self.dom, operands)):

            if free[i]:
                if typeof(operand) not in types:
                    raise TypeError('Signature for %s :: %s does not permit type %s' %
                        (self.__name__, self.signature, typeof(operand)))

            if rigid[i]:
                bound = env.get(var)

                if bound:
                    if typeof(operand) != typeof(bound):
                        uni = unify(operand, bound)
                        if uni in types:
                            env[var] = uni
                else:
                    if typeof(operand) in types:
                        env[var] = typeof(operand)
                    else:
                        raise TypeError('Signature for %s :: %s does not permit type %s' %
                            (self.__name__, self.signature, typeof(operand)))

        # Type checks!
        return env

    @property
    def name(self):
        return self.op


class Slice(Op):
    # $0, start, stop, step
    arity = 4
    opaque = True

#------------------------------------------------------------------------
# Table
#------------------------------------------------------------------------

#------------------------------------------------------------------------
# Values
#------------------------------------------------------------------------

class Literal(Node):
    __slots__ = ['children', 'vtype']

    def __init__(self, val):
        assert isinstance(val, self.vtype)
        self.val = val
        self.children = []

    @property
    def name(self):
        return str(self.val)

class StringNode(Literal):
    vtype = str

#------------------------------------------------------------------------
# Scalars
#------------------------------------------------------------------------

# TODO: more robust!!

class ScalarNode(Literal):
    vtype = int

class IntNode(Literal):
    vtype = int

class DoubleNode(Literal):
    vtype = float

class IndexNode(Literal):
    vtype = tuple

    @property
    def name(self):
        return 'Index%s' % str(self.val)
