import ast
from itertools import repeat

from toolz import merge

from ..core import Expr
from . import numbers
from .core import Scalar

from .interface import ScalarSymbol
from .numbers import (Add, Sub, Mul, Div, And, Or, Eq, NE as Ne, LT as Lt,

                      GT as Gt, LE as Le, GE as Ge, Neg, Pow, Mod)


def generate_methods(node_names, funcs, builder):
    def wrapped(cls):
        for node_name, func in zip(node_names, funcs):
            setattr(cls, 'visit_%s' % node_name, builder(func))
        return cls
    return wrapped


disallowed = ('Attribute', 'Lambda', 'IfExp', 'Dict', 'Set', 'ListComp',
              'SetComp', 'DictComp', 'GeneratorExp', 'Yield')

comparison_ops = {'Eq': Eq, 'Ne': Ne, 'Lt': Lt, 'Gt': Gt, 'Le': Le, 'Ge': Ge,
                  'And': And, 'Or': Or, 'Neg': Neg, 'Add': Add, 'Mult': Mul,
                  'Div': Div, 'Pow': Pow, 'Mod': Mod, 'Sub': Sub}


def disallower(self, node):
    raise NotImplementedError


@generate_methods(disallowed, repeat(disallower, len(disallowed)),
                  builder=lambda func: lambda self, node: func(self, node))
@generate_methods(comparison_ops.keys(), comparison_ops.values(),
                  builder=lambda func: lambda self, node: func)
class BlazeParser(ast.NodeVisitor):
    def __init__(self, dtypes, scope):
        self.dtypes = dtypes
        self.scope = scope

    def visit_Compare(self, node):
        assert len(node.ops) == 1, 'chained comparisons not supported'
        assert len(node.comparators) == 1, 'chained comparisons not supported'
        return self.visit(node.ops[0])(self.visit(node.left),
                                       self.visit(node.comparators[0]))

    def visit_Num(self, node):
        num = node.n
        return ScalarSymbol(num, str(type(num).__name__))

    def visit_Str(self, node):
        return ScalarSymbol(repr(node.s), dtype='string')

    def visit_Name(self, node):
        name = node.id
        if name.startswith('__'):
            raise ValueError("invalid name %r" % name)
        try:
            return self.scope[name]
        except KeyError:
            return ScalarSymbol(name, self.dtypes[name])

    def visit_BinOp(self, node):
        return self.visit(node.op)(self.visit(node.left),
                                   self.visit(node.right))

    def visit_Call(self, node):
        assert len(node.args) <= 1, 'only single argument functions allowed'
        assert not node.keywords
        assert node.starargs is None, 'starargs not allowed'
        assert node.kwargs is None, 'kwargs not allowed'
        return self.visit(node.func)(*map(self.visit, node.args))


# Operations like sin, cos, exp, isnan, floor, ceil, ...
math_operators = dict((k, v) for k, v in numbers.__dict__.items()
                      if isinstance(v, type) and issubclass(v, Scalar))
safe_scope = {'__builtins__': {},  # Python 2
              'builtins': {}}      # Python 3


def exprify(expr, dtypes):
    """ Transform string into scalar expression

    >>> expr = exprify('x + y', {'x': 'int64', 'y': 'real'})
    >>> expr
    x + y
    >>> isinstance(expr, Expr)
    True
    >>> expr.lhs.dshape
    dshape("int64")
    """
    scope = merge(safe_scope, math_operators)

    # use eval mode to raise a SyntaxError if any statements are passed in
    parsed = ast.parse(expr, mode='eval')
    parser = BlazeParser(dtypes, scope)
    return parser.visit(parsed.body)
