from __future__ import absolute_import, division, print_function

import ast

from toolz import merge

from ..core import Expr
from .core import Scalar
from . import numbers
from .parser import BlazeParser


safe_scope = {'__builtins__': {},  # Python 2
              'builtins': {}}      # Python 3
# Operations like sin, cos, exp, isnan, floor, ceil, ...
math_operators = dict((k, v) for k, v in numbers.__dict__.items()
                      if isinstance(v, type) and issubclass(v, Scalar))


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
