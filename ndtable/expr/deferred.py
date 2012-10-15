"""
Core of the deferred expression engine.
"""

from numbers import Number
from collections import Iterable

from ndtable.table import NDTable
from ndtable.expr.nodes import Node, StringNode, ScalarNode

#------------------------------------------------------------------------
# Globals
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
    'repr', 'str', 'hash', 'len', 'abs', 'complex', 'int', 'long', 'float',
    'iter', 'oct', 'hex'
]

PyArray_Intrinsics = [
    "dtype", "size"
]

PyArray_WriteMethods = [
    'fill', 'itemset', 'put'
]

PyArray_ReadMethods = [
    'all', 'any', 'argmax', 'argmin', 'argsort', 'astype', 'base',
    'byteswap', 'choose', 'clip', 'compress', 'conj', 'conjugate',
    'copy', 'ctypes', 'cumprod', 'cumsum', 'data', 'diagonal', 'dot',
    'dtype', 'dump', 'dumps', 'flags', 'flat', 'flatten', 'getfield',
    'imag', 'item', 'itemset', 'itemsize', 'max', 'mean',
    'min', 'nbytes', 'ndim', 'newbyteorder', 'nonzero', 'prod', 'ptp',
    'ravel', 'real', 'repeat', 'reshape', 'resize', 'round',
    'searchsorted', 'setasflat', 'setfield', 'setflags', 'shape',
    'size', 'sort', 'squeeze', 'std', 'strides', 'sum', 'swapaxes',
    'take', 'tofile', 'tolist', 'tostring', 'trace', 'transpose', 'var',
    'view'
]

#------------------------------------------------------------------------
# Graph Construction
#------------------------------------------------------------------------

def is_homogeneous(it):
    # type() comparisions are cheap pointer arithmetic on
    # PyObject->tp_type, isinstance() calls are expensive since
    # they have travese the whole mro hierarchy

    head = it[0]
    head_type = type(head)
    return head, [type(a) == head_type for a in it]

def injest_iterable(args, depth=0):
    # TODO: Should be 1 stack frame per each recursion so we
    # don't blow up Python trying to parse big structures

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
            is_hetero = not is_homog

            # Homogenous Arguments
            # ====================

            if is_homog:
                if isinstance(head, Number):
                    return [ScalarNode(a) for a in args]
                elif isinstance(head, basestring):
                    return [StringNode(a) for a in args]
                elif isinstance(head, DeferredTable):
                    return [a.node for a in args]
                else:
                    return args

            # Heterogenous Arguments
            # ======================

            elif is_hetero:
                args = []
                for a in args:
                    if isinstance(a, Iterable):
                        sub = injest_iterable(a, depth+1)
                        a.append(sub)
                    elif isinstance(a, DeferredTable):
                        args.append(a.node)
                    elif isinstance(a, Node):
                        args.append(a)
                    elif isinstance(a, Number):
                        args.append(ScalarNode(a))
                    elif isinstance(a, basestring):
                        args.append(StringNode(a))
                    else:
                        raise TypeError("Unknown type")
                return args

        else:
            raise RuntimeError("""
            "Too many dynamic arguments to build expression
            graph. Consider alternative construction.""")

#------------------------------------------------------------------------
# Table
#------------------------------------------------------------------------

# A thin wrapper around a Node object
class DeferredTable(object):

    def __init__(self, args, target=NDTable, depends=None):
        # We want the operations on the table to be
        # closed ( in the algebraic sense ) so that we don't
        # escape to Tables when we're using DataTables.
        self.target = target

        # Defer the arguments until we injest them

        # Auto resolve the graph

        if depends is None:
            fields = injest_iterable(args)
            self.node = Node('init', *fields)
            self.node.depends_on(fields)
        else:
            self.node = Node('init', args, target)
            self.node.depends_on(*depends)

    def generate_node(self, fname, args, kwargs):
        return Node(fname, args, kwargs)

    # Read Operations
    # ===============

    # Python Intrinsics
    # -----------------
    for name in PyObject_Intrinsics:
        exec (
            "def __%(name)s__(self,*args, **kwargs):\n"
            "    return self.generate_node('%(name)s',args, kwargs)"
        ) % locals()

    # Unary Prefix
    # ------------
    for name, op in PyObject_UnaryOperators:
        exec (
            "def __%(name)s__(self):\n"
            "    return self.generate_node('%(name)s',args, kwargs)"
        ) % locals()

    for name in PyArray_ReadMethods:
        exec (
            "def %(name)s(self, *args, **kwargs):\n"
            "    return self.generate_node('%(name)s',args, kwargs)"
        ) % locals()

    # Binary Prefix
    # -------------
    for name, op in PyObject_BinaryOperators:
        exec (
            "def __%(name)s__(self,ob):\n"
            "    return self.generate_node('%(name)s',args, kwargs)"
            "\n"
            "def __r%(name)s__(self,ob):\n"
            "    return self.generate_node('%(name)s',args, kwargs)"
            "\n"
        )  % locals()

    # Write Operations
    # ===============

    for name, op in PyObject_BinaryOperators:
        exec (
            "def __i%(name)s__(self,ob):\n"
            "    return self.generate_node('%(name)s',args, kwargs)"
            "\n"
        )  % locals()

    for name in PyArray_WriteMethods:
        exec (
            "def %(name)s(self, *args, **kwargs):\n"
            "    return self.generate_node('%(name)s',args, kwargs)"
        ) % locals()
