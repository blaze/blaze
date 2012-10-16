"""
Core of the deferred expression engine.
"""

from collections import Iterable
from numbers import Number, Integral

from ndtable.table import NDTable
from ndtable.expr.nodes import Node, StringNode, ScalarNode,\
    Slice, NullaryOp, UnaryOp, BinaryOp, NaryOp, Op, IndexNode

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
    'str', 'hash', 'len', 'abs', 'complex', 'int', 'long', 'float',
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

#------------------------------------------------------------------------
# Graph Construction
#------------------------------------------------------------------------

def is_homogeneous(it):
    # type() comparisions are cheap pointer arithmetic on
    # PyObject->tp_type, isinstance() calls are expensive since
    # they have travese the whole mro hierarchy

    head = it[0]
    head_type = type(head)
    return head, all([type(a) == head_type for a in it])

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

            # TODO: This will be really really slow, certainly
            # not something we'd want to put in a loop.
            # Optimize later!

            elif is_hetero:
                ret = []
                for a in args:
                    if isinstance(a, (list, tuple)):
                        sub = injest_iterable(a, depth+1)
                        ret.append(sub)
                    elif isinstance(a, DeferredTable):
                        ret.append(a.node)
                    elif isinstance(a, Node):
                        ret.append(a)
                    elif isinstance(a, Number):
                        ret.append(ScalarNode(a))
                    elif isinstance(a, basestring):
                        ret.append(StringNode(a))
                    else:
                        raise TypeError("Unknown type")
                return ret

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

        if depends is None:
            self.node = self
            self.children = injest_iterable(args)
        else:
            self.node = self
            self.children = depends

    def generate_node(self, arity, fname, args=None, kwargs=None):

        # TODO: also kwargs when we support such things
        iargs = injest_iterable(args)

        if arity == -1:
            ret = NaryOp(fname, iargs, kwargs)

        elif arity == 0:
            assert len(iargs) == arity
            ret =  NullaryOp(fname)

        elif arity == 1:
            assert len(iargs) == arity
            ret =  UnaryOp(fname, iargs)

        elif arity == 2:
            assert len(iargs) == arity
            ret = BinaryOp(fname, iargs)

        return ret

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
        return Op('transpose', self.node)

    # Read Operations
    # ===============

    def __getitem__(self, idx):
        """ Slicing operations should return graph nodes, while individual
        element access should return bare scalars.
        """
        # TODO: how to represent indexer in graph???

        if isinstance(idx, Integral) or isinstance(idx, np.integer):
            ndx = IndexNode((idx,))
            return Slice('getitem', [self.node, ndx])
        else:
            ndx = IndexNode(idx)
            return Slice('getitem', [self.node, ndx])

    def __getslice__(self, start, stop, step):
        """
        """
        ndx = IndexNode((start, stop, step))
        return Slice(self.node, ndx)

    # Python Intrinsics
    # -----------------
    for name in PyObject_Intrinsics:
        # Bound methods are actually just unary functions with
        # the first argument self
        exec (
            "def __%(name)s__(self):\n"
            "    return self.generate_node(1, '%(name)s', [self.node])"
            "\n"
        ) % locals()

    # Unary Prefix
    # ------------
    for name, op in PyObject_UnaryOperators:
        exec (
            "def __%(name)s__(self):\n"
            "    return self.generate_node(1, '%(name)s', [self.node])"
            "\n"
        ) % locals()

    for name in PyArray_ReadMethods:
        exec (
            "def %(name)s(self, *args, **kwargs):\n"
            "    args = (self.node,) + args\n"
            "    return self.generate_node(-1, '%(name)s', args, kwargs)"
            "\n"
        ) % locals()

    # Binary Prefix
    # -------------
    for name, op in PyObject_BinaryOperators:
        exec (
            "def __%(name)s__(self, ob):\n"
            "    return self.generate_node(2, '%(name)s', [self.node, ob])\n"
            "\n"
            "def __r%(name)s__(self, ob):\n"
            "    return self.generate_node(2, '%(name)s', [self.node, ob])\n"
            "\n"
        )  % locals()

    # Write Operations
    # ===============

    for name, op in PyObject_BinaryOperators:
        exec (
            "def __i%(name)s__(self, ob):\n"
            "    return self.generate_node(2, '%(name)s', [self.node, ob])\n"
            "\n"
        )  % locals()

    for name in PyArray_WriteMethods:
        exec (
            "def %(name)s(self, *args, **kwargs):\n"
            "    return self.generate_node(-1, '%(name)s', args, kwargs)\n"
            "\n"
        ) % locals()

    @property
    def name(self):
        return self.debug_name() # for now

    def debug_name(self):
        # unholy magic, just for debugging! Returns the variable
        # name of the object assigned to
        #
        #    a = DeferredTable()
        #    a.name == 'a'

        import sys, gc

        def find_names(obj):
            frame = sys._getframe(1)
            while frame is not None:
                frame.f_locals
                frame = frame.f_back

            for referrer in gc.get_referrers(obj):
                if isinstance(referrer, dict):
                    for k, v in referrer.iteritems():
                        if v is obj:
                            if len(k) == 1:
                                return k

        return find_names(self)

    # Other non-graph methods
    # ========================

    def tofile(self, *args, **kw):
        pass

    def tolist(self, *args, **kw):
        pass

    def tostring(self, *args, **kw):
        pass
