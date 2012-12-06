"""
Deprecating this in favor of parser.py
"""

import imp
import ast
import inspect
from operator import add
from string import maketrans, translate
from collections import OrderedDict, Iterable

from coretypes import Integer, TypeVar, Record, Enum, Type, DataShape, \
    Range, Either, Fixed

syntax_error = """
  File {filename}, line {lineno}
    {line}
    {pointer}

DatashapeSyntaxError: {msg}
"""

class DatashapeSyntaxError(Exception):
    """
    Makes datashape parse errors look like Python SyntaxError.
    """
    def __init__(self, filename, lineno, col_offset, text, msg=None):
        self.lineno     = lineno
        self.col_offset = col_offset
        self.filename   = filename
        self.text       = text
        self.msg        = msg or 'invalid syntax'

    def __str__(self):
        return syntax_error.format(**{
            'filename' : self.filename,
            'lineno'   : self.lineno,
            'line'     : self.text,
            'pointer'  : ' '*self.col_offset + '^',
            'msg'      : self.msg
        })

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
            return DatashapeSyntaxError('<stdin>', ast_node.lineno, ast_node.col_offset, line)
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

    def Call(self, tree):
        # TODO: don't inline this
        internals = {
            'Record'   : Record,
            'Enum'     : Enum,
            'Range'    : Range,
            'Either'   : Either,
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
        return Enum(*args)

    def Index(self, tree):
        return self.visit(tree.value)

class TranslateModule(Translate):

    def Module(self, tree):
        return [self.visit(i) for i in tree.body]

    def Assign(self, tree):
        left = tree.targets[0].id
        right = self.visit(tree.value)
        assert left not in self.namespace
        self.namespace[left] = right


#------------------------------------------------------------------------
# Operator Translation
#------------------------------------------------------------------------

def parse(expr):
    expr_translator = Translate(expr)
    try:
        past = ast.parse(expr, '<string>', mode='eval')
    except SyntaxError as e:
        raise DatashapeSyntaxError(*e.args[1])
    return expr_translator.visit(past)

def load(fname, modname=None):
    """
    # Load a file of datashape definitions as if it were a python module.
    """

    with open(fname, 'r') as fd:
        expr = fd.read()
        expr = translate(expr, op_table)
        past = ast.parse(expr)

    translator = TranslateModule(expr)
    translator.visit(past)

    if not modname:
        modname = fname
    mod = imp.new_module(modname)

    for k,v in translator.namespace.iteritems():
        setattr(mod, k, v)
    return mod
