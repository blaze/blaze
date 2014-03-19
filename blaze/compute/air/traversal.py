"""Visitor and transformer helpers.

    transform(transformer, func):
        transform Ops in func using transformer

    visit(visitor, func):
        visit Ops in func

    vvisit(visitor, func):
        visit Ops in func and track values for each Op, returned
        by each visit method

    Combinator([visitors...]):
        Combine a bunch of visitors into one
"""

from __future__ import print_function, division, absolute_import
import inspect

from .utils import nestedmap
from .error import CompileError


def _missing(visitor, op):
    raise CompileError(
                "Opcode %r not implemented by %s" % (op.opcode, visitor))


def transform(obj, function, handlers=None, errmissing=False):
    """Transform a bunch of operations"""
    obj = combine(obj, handlers)
    for op in function.ops:
        fn = getattr(obj, 'op_' + op.opcode, None)
        if fn is not None:
            result = fn(op)
            if result is not None and result is not op:
                op.replace(result)
        elif errmissing:
            _missing(obj, op)


def visit(obj, function, handlers=None, errmissing=False):
    """Visit a bunch of operations"""
    obj = combine(obj, handlers)
    for op in function.ops:
        fn = getattr(obj, 'op_' + op.opcode, None)
        if fn is not None:
            fn(op)
        elif errmissing:
            _missing(obj, op)


def vvisit(obj, function, argloader=None, valuemap=None, errmissing=True):
    """
    Visit a bunch of operations and track values. Uses ArgLoader to
    resolve Op arguments.
    """
    argloader = argloader or ArgLoader()
    valuemap = argloader.store if valuemap is None else valuemap

    for arg in function.args:
        valuemap[arg.result] = obj.op_arg(arg)

    for block in function.blocks:
        obj.blockswitch(argloader.load_Block(block))
        for op in block.ops:
            fn = getattr(obj, 'op_' + op.opcode, None)
            if fn is not None:
                args = argloader.load_args(op)
                result = fn(op, *args)
                valuemap[op.result] = result
            elif errmissing:
                _missing(obj, op)

    return valuemap


class ArgLoader(object):
    """
    Resolve Operation values and Operation arguments. This keeps a store that
    can be used for translation or interpretation, mapping IR values to
    translation or runtime values (e.g. LLVM or Python values).

        store: { Value : Result }
    """

    def __init__(self, store=None):
        self.store = store if store is not None else {}

    def load_op(self, op):
        from pykit.ir import Value, Op

        if isinstance(op, Value):
            return getattr(self, 'load_' + type(op).__name__)(op)
        else:
            return op

    def load_args(self, op):
        if op.opcode == 'phi':
            # phis have cycles and values cannot be loaded in a single pass
            return ()
        return nestedmap(self.load_op, op.args)

    def load_Block(self, arg):
        return arg

    def load_Constant(self, arg):
        return arg.const

    def load_Pointer(self, arg):
        return arg

    def load_Struct(self, arg):
        return arg

    def load_GlobalValue(self, arg):
        raise NotImplementedError

    def load_Function(self, arg):
        return arg

    def load_Operation(self, arg):
        if arg.result not in self.store:
            raise NameError("%s not in %s" % (arg, self.store))
        return self.store[arg.result]

    load_FuncArg = load_Operation


class Combinator(object):
    """
    Combine several visitors/transformers into one.
    One can also use dicts wrapped in pykit.utils.ValueDict.
    """

    def __init__(self, visitors, prefix='op_', index=None):
        self.visitors = visitors
        self.index = _build_index(visitors, prefix)
        if index:
            assert not set(index) & set(self.index)
            self.index.update(index)

    def __getattr__(self, attr):
        try:
            return self.index[attr]
        except KeyError:
            if len(self.visitors) == 1:
                # no ambiguity
                return getattr(self.visitors[0], attr)
            raise AttributeError(attr)


def _build_index(visitors, prefix):
    """Build a method table of method names starting with `prefix`"""
    index = {}
    for visitor in visitors:
        for attr, method in inspect.getmembers(visitor):
            if attr.startswith(prefix):
                if attr in index:
                    raise ValueError("Handler %s not unique!" % attr)
                index[attr] = method

    return index


def combine(visitor, handlers):
    """Combine a visitor/transformer with a handler dict ({'name': func})"""
    if handlers:
        visitor = Combinator([visitor], index=handlers)
    return visitor
