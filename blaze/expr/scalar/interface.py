from __future__ import absolute_import, division, print_function

import numbers as pynumbers
from datetime import datetime, date
from ..core import Expr
from datashape import dshape, DataShape, Function
from .boolean import BooleanInterface
from .numbers import NumberInterface, RealMath, dispatch, BinOp, UnaryOp
from .numbers import IntegerMath, BooleanMath
import pandas as pd
from cytoolz import merge, keyfilter
import math
from blaze.compatibility import basestring


class ScalarSymbol(NumberInterface, BooleanInterface):
    __slots__ = '_name', 'dtype'
    __inputs__ = ()

    def __init__(self, name, dtype='real'):
        self._name = name
        self.dtype = dshape(dtype)

    @property
    def name(self):
        return self._name

    @property
    def dshape(self):
        return dshape(self.dtype)

    def __str__(self):
        return str(self._name)

    __hash__ = Expr.__hash__


class Expressify(object):
    def __init__(self, scope):
        self.scope = scope

    @dispatch((RealMath, IntegerMath, BooleanMath))
    def visit(self, node):
        f = self.scope[node.symbol]
        return f(*map(self.visit, node.args))

    @dispatch(ScalarSymbol)
    def visit(self, node):
        return self.scope[node.name]

    @dispatch((pynumbers.Number, datetime, date))
    def visit(self, node):
        return node

    @dispatch(basestring)
    def visit(self, s):
        # dateutil accepts the empty string as a valid datetime, don't let it do
        # that
        if s:
            try:
                return pd.Timestamp(s).to_pydatetime()
            except ValueError:
                return s
        return s

    @dispatch(BinOp)
    def visit(self, node):
        return node.op(self.visit(node.lhs), self.visit(node.rhs))

    @dispatch(UnaryOp)
    def visit(self, node):
        return node.op(self.visit(node.child))


class Lambda(Expr):

    __slots__ = 'child', 'expr'

    __default_scope__ = keyfilter(lambda x: not x.startswith('__'),
                                  math.__dict__)

    @property
    def columns(self):
        return list(map(str, self.child.columns))

    @property
    def dshape(self):
        restype = self.expr.dshape
        argtypes = tuple(map(dshape, self.child.schema[0].types))
        types = argtypes + (restype,)
        return DataShape(Function(*types))

    def __repr__(self):
        return 'lambda (%s): %s' % (', '.join(self.columns), self.expr)

    def __call__(self, row):
        scope = merge(self.__default_scope__, dict(zip(self.columns, row)))
        parser = Expressify(scope)
        return parser.visit(self.expr)
