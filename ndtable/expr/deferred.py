"""
Core of the deferred expression engine.
"""

from numbers import Number
from collections import Iterable

from ndtable.table import DataTable
from ndtable.expr.nodes import Node, StringNode, ScalarNode, Slice


# conditional import of Numpy; if it doesn't exist, then set up dummy objects
# for the things we use
try:
    import numpy as np
except ImportError:
    np = {"integer": numbers.Integral}

_max_argument_len       = 1000
_max_argument_recursion = 25
_argument_sample        = 100

def set_max_argument_len(val):
    global _max_argument_len
    _max_argument_len = val

def set_max_argument_recursion(val):
    global _max_argument_recursion
    _max_argument_recursion = val

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
    'repr', 'str', 'hash', 'len', 'abs', 'complex', 'int', 'long', 'float',
    'iter', 'oct', 'hex'
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


def is_homogeneous(it):
    # type() comparisions are cheap pointer arithmetic on
    # PyObject->tp_type, isinstance() calls are expensive since
    # they have travese the whole mro hierarchy

    head = type(it[0])
    return head, [type(a) == head for a in it]

def injest_iterable(graph, arg, depth=0):
    # TODO: Should be 1 stack frame per each recursion so we
    # don't blow up Python trying to parse big structures

    if depth > _max_argument_recursion:
        raise RuntimeError(\
        "Maximum recursion depth reached while parsing arguments")

    # tuple, list, dictionary, any recursive combination of them
    if isinstance(arg, Iterable):
        if len(arg) < _max_argument_len:
            sample = arg[0:_argument_sample]

            # If the first 100 elements are type homogenous then
            # it's likely the rest of the iterable is.
            head, is_homog = is_homogeneous(sample)
            is_hetero = not is_homog

            if is_homog and isinstance(head, Number):
                return [ScalarNode(graph, a) for a in arg]
            elif is_homog and isinstance(head, basestring):
                return [StringNode(graph, a) for a in arg]

            # Worst-case, we have a heterogenous list of
            if is_hetero:
                args = []
                for a in arg:
                    if isinstance(a, Iterable):
                        sub = injest_iterable(graph, a, depth+1)
                        a.append(sub)
                    elif isinstance(a, Node):
                        args.append(a)
                    elif isinstance(a, Number):
                        args.append(ScalarNode(graph, a))
                    elif isinstance(a, basestring):
                        args.append(StringNode(graph, a))
                    else:
                        raise TypeError("Unknown type")
                return args

        else:
            raise RuntimeError("""
            "Too many dynamic arguments to build expression
            graph. Consider alternative construction.""")

class DeferredTable(object):

    def __init__(self, source, target=DataTable, depends=None):
        # We want the operations on the table to be
        # closed ( in the algebraic sense ) so that we don't
        # escape to Tables when we're using DataTables.
        self.target = target

        # Defer the arguments until we injest them

        # Auto resolve the graph

        if depends is None:
            self.node = Node('__init__', **{
                'args'   : injest_iterable(self.node),
                'target' : target,
            })

        else:
            self.node = Node('__init__', source, target)
            self.node.depends_on(depends)

    def generate_node(self, fname, args, kwargs):
        return Node(fname, args, kwargs)


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

    # Numpy-compatible data attributes
    # ================================

    @property
    def imag(self):
        # TODO: Add an imag() graph operator, and make this call it.
        pass

    @property
    def real(self):
        # TODO: Add a real() graph operator, and make this call it.
        pass

    @property
    def flat(self):
        """ Equivalent to .reshape(), which returns a graph node. """
        pass

    @property
    def T(self):
        """ Equivalent to .transpose(), which returns a graph node. """
        pass


    # Read Operations
    # ===============

    def __getitem__(self, ndx):
        """ Slicing operations should return graph nodes, while individual
        element access should return bare scalars.
        """
        if isinstance(ndx, numbers.Integral) or isinstance(ndx, np.integer):
            return self._call("__getitem__", (ndx,), {})
        else:
            return self.generate_node(Slice, ndx)

    # Python Intrinsics
    # -----------------
    for name in PyObject_Intrinsics:
        exec (
            "def __%(name)s__(self,*args, **kwargs):\n"
            "    return self.generate_node(%(name)s args, kwargs)"
        ) % locals()

    # Unary Prefix
    # ------------
    for name, op in PyObject_UnaryOperators:
        exec (
            "def __%(name)s__(self):\n"
            "    return self.generate_node(%(name)s args, kwargs)"
        ) % locals()

    for name in PyArray_ReadMethods:
        exec (
            "def %(name)s(self, *args, **kwargs):\n"
            "    return self.generate_node(%(name)s args, kwargs)"
        ) % locals()

    # Binary Prefix
    # -------------
    for name, op in PyObject_BinaryOperators:
        exec (
            "def __%(name)s__(self,ob):\n"
            "    return self.generate_node(%(name)s args, kwargs)"
            "\n"
            "def __r%(name)s__(self,ob):\n"
            "    return self.generate_node(%(name)s args, kwargs)"
            "\n"
        )  % locals()

    # Write Operations
    # ===============

    for name, op in PyObject_BinaryOperators:
        exec (
            "def __i%(name)s__(self,ob):\n"
            "    return self.generate_node(%(name)s args, kwargs)"
            "\n"
        )  % locals()

    for name in PyArray_WriteMethods:
        exec (
            "def %(name)s(self, *args, **kwargs):\n"
            "    return self.generate_node(%(name)s args, kwargs)"
        ) % locals()

    # Other non-graph methods
    # ========================

    def tofile(self, *args, **kw):
        pass

    def tolist(self, *args, **kw):
        pass

    def tostring(self, *args, **kw):
        pass

