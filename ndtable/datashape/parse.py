"""
Parser for DataShape grammer.
"""

import re
import ast
import inspect
from collections import OrderedDict, Iterable
from datashape import Integer, TypeVar, Tuple, Record, Function, \
    Enum, Type, DataShape, Stream, Var

class Visitor(object):

    def __init__(self):
        super(Visitor,self).__init__()
        self.fallback = self.Unknown

    def Unknown(self, tree):
        raise SyntaxError()

    def visit(self, tree):
        if isinstance(tree, Iterable):
            return [self.visit(i) for i in tree]
        else:
            nodei = tree.__class__.__name__
            if nodei in self.__class__.__dict__:
                return getattr(self,nodei)(tree)
            else:
                return self.Unknown(tree)

class Translate(Visitor):
    """
    Translate PyAST to DataShape types
    """

    def Expression(self, tree):
        operands = self.visit(tree.body),
        return DataShape(operands)

    # list -> Enum
    def List(self, tree):
        args = map(self.visit, tree.elts)
        return Enum(*args)

    # int -> Integer
    def Num(self, tree):
        if isinstance(tree.n, int):
            return Integer(tree.n)
        else:
            raise ValueError()

    # var -> TypeVar
    def Name(self, tree):
        if tree.id in Type.registry:
            return Type.registry[tree.id]
        else:
            return TypeVar(tree.id)

    # (f . g) -> Compose
    def Attribute(self, tree):
        # Would be a composition operator ( f . g )
        #a = self.visit(tree.value)
        #b = tree.attr
        raise NotImplementedError()

    # function call -> Function
    def Call(self, tree):
        internals = {
            'Record' : Record,
            'Enum'   : Enum,
            'Stream' : Stream,
            'Var'    : Var,
        }
        internals.update(Type.registry)

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


    # tuple -> DataShape
    def Tuple(self, tree):
        operands = self.visit(tree.elts)
        return DataShape(operands)

    def Index(self, tree):
        return self.visit(tree.value)

    # " (a,b) -> c " -> Function((a,b), c)
    def BinOp(self, tree):
        if type(tree.op) is ast.RShift:
            left = self.visit(tree.left)
            right = self.visit(tree.right)
            return Function(arg_type=left, ret_type=right)
        else:
            raise SyntaxError()

    def Subscript(self, tree):
        raise NotImplementedError()

    def Slice(self, tree):
        raise NotImplementedError()

    def Lambda(self, tree):
        raise NotImplementedError()

translator = Translate()

def parse(expr):
    expr = re.sub(r'->', '>>', expr)
    past = ast.parse(expr, '<string>', mode='eval')
    return translator.visit(past)
