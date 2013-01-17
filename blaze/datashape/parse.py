"""
WARNING: Deprecating this in favor of parser.py
"""

import ast
import inspect
from collections import OrderedDict, Iterable

from coretypes import Integer, TypeVar, Record, Factor, Type, DataShape, \
    Range, Either, Fixed, Varchar, String

from blaze.error import CustomSyntaxError

syntax_error = """
  File {filename}, line {lineno}
    {line}
    {pointer}

DatashapeSyntaxError: {msg}
"""

class DatashapeSyntaxError(CustomSyntaxError):
    pass

class Visitor(object):

    def __init__(self, source = None):
        self.namespace = {}
        self.source = source

    def visit(self, tree):
        if isinstance(tree, Iterable):
            return [self.visit(i) for i in tree]
        else:
            nodei = tree.__class__.__name__
            trans = getattr(self,nodei, False)
            if trans:
                return trans(tree)
            else:
                return self.Unknown(tree)

    def Unknown(self, tree):
        raise self.error_ast(tree)

    def error_ast(self, ast_node):
        if self.source:
            line = self.source.split('\n')[ast_node.lineno - 1]
            return DatashapeSyntaxError('<stdin>', ast_node.lineno,
                    ast_node.col_offset, line)
        else:
            raise SyntaxError()

class Translate(Visitor):
    """
    Translate PyAST to DataShape types
    """

    # tuple -> DataShape
    def Tuple(self, tree):
        def toplevel(elt):
            if isinstance(elt, ast.Num):
                return Fixed(elt.n)
            else:
                return self.visit(elt)
        operands = map(toplevel, tree.elts)
        return DataShape(operands)

    def Expression(self, tree):
        operands = self.visit(tree.body),
        return DataShape(operands)

    # int -> Integer
    def Num(self, tree):
        if isinstance(tree.n, int):
            return Integer(tree.n)
        else:
            raise ValueError()

    # var -> TypeVar
    def Name(self, tree):
        if tree.id in Type._registry:
            return Type._registry[tree.id]
        else:
            return TypeVar(tree.id)

    def Dict(self, tree):
        k = [k.id for k in tree.keys]
        v = map(self.visit, tree.values)

        return Record(zip(k,v))

    def Call(self, tree):
        # TODO: don't inline this
        internals = {
            'Record'   : Record,
            'Range'    : Range,
            'Either'   : Either,
            'Varchar'  : Varchar,
            'String'   : String,
        }
        internals.update(Type._registry)

        fn = self.visit(tree.func)
        args = self.visit(tree.args)

        k, v = [], []

        for kw in tree.keywords:
            k += [kw.arg]
            v += [self.visit(kw.value)]

        kwargs = OrderedDict(zip(k,v))

        if type(fn) is TypeVar and fn.symbol in internals:
            try:
                return internals[fn.symbol](*args, **kwargs)
            except TypeError:
                n = len(inspect.getargspec(internals[fn.symbol].__init__).args)
                m = len(args)
                raise Exception('Constructor %s expected %i arguments, got %i' % \
                    (fn.symbol, (n-1), m))
        else:
            raise NameError(fn.symbol)


    def Set(self, tree):
        args = map(self.visit, tree.elts)
        return Factor(*args)

    def Index(self, tree):
        return self.visit(tree.value)


def parse(expr):
    expr_translator = Translate(expr)
    try:
        past = ast.parse(expr, '<string>', mode='eval')
    except SyntaxError as e:
        raise DatashapeSyntaxError(*e.args[1])
    return expr_translator.visit(past)
