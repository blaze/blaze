"""
Core of the deferred expression engine.
"""

from functools import wraps
from ndtable.table import DataTable

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

def tablefunction(fn, nin, nout):
    """
    Wrap a arbitrary array function into a deferredk table function.
    """
    @wraps(fn)
    def wrapper(self):
        return Function(fn, nin, nout)
    return wrapper

class DeferredTable(object):

    def __init__(self, source, target=DataTable):
        self._underlying = source
        # We want the operations on the table to be
        # closed ( in the algebraic sense ) so that we don't
        # escape to Tables when we're using DataTables.
        self.target = target

    def __unsafe_peek__(self):
        """
        Peek at the underlying array, for debugging.
        """
        return self._underlying

    @property
    def size(self):
        return self._underlying.size

    @property
    def dtype(self):
        return self._underlying.dtype

    # Read Operations
    # ===============

    # Python Intrinsics
    # -----------------
    for name in PyObject_Intrinsics:
        exec (
            "def __%(name)s__(self,*args, **kwargs):\n"
            "    with self.reading:"
            "        return self._underlying.__%(name)s__()"
        ) % locals()

    # Unary Prefix
    # ------------
    for name, op in PyObject_UnaryOperators:
        exec (
            "def __%(name)s__(self):\n"
            "    with self.reading:\n"
            "        return self._underlying.__%(name)s__()"
        ) % locals()

    for name in PyArray_ReadMethods:
        exec (
            "def %(name)s(self, *args, **kwargs):\n"
            "    with self.reading:\n"
            "        return self._underlying.%(name)s(*args, **kwargs)"
        ) % locals()

    # Binary Prefix
    # -------------
    for name, op in PyObject_BinaryOperators:
        exec (
            "def __%(name)s__(self,ob):\n"
            "    with self.reading:"
            "        return self._underlying %(op)s ob\n"
            "\n"
            "def __r%(name)s__(self,ob):\n"
            "    with self.reading:"
            "        return ob %(op)s self._underlying\n"
            "\n"
        )  % locals()

    # Write Operations
    # ===============

    for name, op in PyObject_BinaryOperators:
        exec (
            "def __i%(name)s__(self,ob):\n"
            "    with self.writing:"
            "        return ob %(op)s self._underlying\n"
            "\n"
        )  % locals()

    for name in PyArray_WriteMethods:
        exec (
            "def %(name)s(self, *args, **kwargs):\n"
            "    with self.writing:\n"
            "        return self._underlying.%(name)s(*args, **kwargs)"
        ) % locals()
