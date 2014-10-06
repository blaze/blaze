from __future__ import absolute_import, division, print_function

from datashape.predicates import iscollection, isscalar
from toolz import partial, unique, first
import datashape
from datashape import dshape, DataShape, Record, Var, Option, Unit

from .expressions import ElemWise, Label, Expr, Symbol, Field
from .core import eval_str
from .arithmetic import (Eq, Ne, Lt, Le, Gt, Ge, Add, Mult, Div, Sub, Pow, Mod,
                         Or, And, USub, Not, FloorDiv)
from . import math

__all__ = ['broadcast', 'Broadcast']

def _expr_child(col):
    """ Expr and child of field

    Examples
    --------

    >>> from blaze import TableSymbol
    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> _expr_child(accounts['name'])
    (name, accounts)

    Helper function for ``broadcast``
    """
    if isinstance(col, (Broadcast, Field)):
        return col.expr, col.child
    elif isinstance(col, Label):
        return _expr_child(col.child)
    else:
        return col, None


def broadcast(op, *column_inputs):
    """ Broadcast scalar operation across multiple fields

    Parameters
    ----------
    op : Scalar Operation like Add, Mult, Sin, Exp

    column_inputs : either Column, Broadcast or constant (like 1, 1.0, '1')

    Examples
    --------

    >>> from blaze import TableSymbol
    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')

    >>> broadcast(Add, accounts['amount'], 100)
    accounts['amount'] + 100

    Fuses operations down into ScalarExpr level

    >>> broadcast(Mult, 2, (accounts['amount'] + 100))
    2 * (accounts['amount'] + 100)
    """
    expr_inputs = []
    children = set()

    for col in column_inputs:
        expr, child = _expr_child(col)
        expr_inputs.append(expr)
        if child:
            children.add(child)

    if not len(children) == 1:
        raise ValueError("All inputs must be from same Table.\n"
                         "Saw the following tables: %s"
                         % ', '.join(map(str, children)))

    if hasattr(op, 'op'):
        expr = op.op(*expr_inputs)
    else:
        expr = op(*expr_inputs)

    return Broadcast(first(children), expr)

class Broadcast(ElemWise):
    """ Apply Scalar Expression onto columns of data

    Parameters
    ----------

    child : TableExpr
    expr : ScalarExpr
        The names of the varibles within the scalar expr must match the columns
        of the child.  Use ``Column.scalar_variable`` to generate the
        appropriate scalar Symbol

    Examples
    --------

    >>> from blaze import TableSymbol, Add
    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')

    >>> expr = Add(accounts['amount'].expr, 100)
    >>> Broadcast(accounts, expr)
    accounts['amount'] + 100

    See Also
    --------

    blaze.expr.table.broadcast
    """
    __slots__ = 'child', 'expr'

    @property
    def _name(self):
        names = [x._name for x in self.expr._traverse()
                  if isinstance(x, Symbol)]
        if len(names) == 1 and not isinstance(self.expr.dshape[0], Record):
            return names[0]

    @property
    def dshape(self):
        return DataShape(*(self.child.shape + (self.expr.dshape.measure,)))

    def __str__(self):
        columns = self.active_columns()
        newcol = lambda c: "%s['%s']" % (self.child, c)
        return eval_str(self.expr.subs(dict(zip(columns,
                                                map(newcol, columns)))))

    def active_columns(self):
        return sorted(unique(x._name for x in self._traverse()
                   if isinstance(x, Symbol) and isscalar(x.dshape)))


def _eq(self, other):
    if (isscalar(self.dshape.measure) and
            (not isinstance(other, Expr)
                 or isscalar(other.dshape.measure))):
        return broadcast(Eq, self, other)
    else:
        return self.isidentical(other)

def _ne(a, b):
    return broadcast(Ne, a, b)

def _lt(a, b):
    return broadcast(Lt, a, b)

def _le(a, b):
    return broadcast(Le, a, b)

def _gt(a, b):
    return broadcast(Gt, a, b)

def _ge(a, b):
    return broadcast(Ge, a, b)

def _add(a, b):
    return broadcast(Add, a, b)

def _radd(a, b):
    return broadcast(Add, b, a)

def _mul(a, b):
    return broadcast(Mult, a, b)

def _rmul(a, b):
    return broadcast(Mult, b, a)

def _div(a, b):
    return broadcast(Div, a, b)

def _rdiv(a, b):
    return broadcast(Div, b, a)

def _floordiv(a, b):
    return broadcast(FloorDiv, a, b)

def _rfloordiv(a, b):
    return broadcast(FloorDiv, b, a)

def _sub(a, b):
    return broadcast(Sub, a, b)

def _rsub(a, b):
    return broadcast(Sub, b, a)

def _pow(a, b):
    return broadcast(Pow, a, b)

def _rpow(a, b):
    return broadcast(Pow, b, a)

def _mod(a, b):
    return broadcast(Mod, a, b)

def _rmod(a, b):
    return broadcast(Mod, b, a)

def _or(a, b):
    return broadcast(Or, a, b)

def _ror(a, b):
    return broadcast(Or, b, a)

def _and(a, b):
    return broadcast(And, a, b)

def _rand(a, b):
    return broadcast(And, b, a)

def _neg(a):
    return broadcast(USub, a)

def _invert(a):
    return broadcast(Not, a)

def isnan(expr):
    return broadcast(math.isnan, expr)

from .expressions import dshape_method_list

dshape_method_list.extend([
    (lambda ds: iscollection(ds) and isscalar(ds.measure),
            set([_eq, _ne, _lt, _le, _gt, _ge, _add, _radd, _mul,
                 _rmul, _div, _rdiv, _floordiv, _rfloordiv, _sub, _rsub, _pow,
                _rpow, _mod, _rmod, _or, _ror, _and, _rand, _neg, _invert]))
    ])
