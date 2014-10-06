""" An abstract Table

>>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
>>> deadbeats = accounts['name'][accounts['amount'] < 0]
"""
from __future__ import absolute_import, division, print_function

from datashape import dshape, DataShape, Record, isdimension, Option
from datashape import coretypes as ct
import datashape
import toolz
from toolz import (concat, partial, first, compose, get, unique, second,
                   isdistinct, frequencies, memoize)
from datashape.predicates import isscalar, iscollection
import numpy as np
from .core import *
from .expressions import *
from .collections import *
from .reductions import *
from .broadcast import broadcast, Broadcast
from ..compatibility import _strtypes, builtins, unicode, basestring, map, zip
from ..dispatch import dispatch


from .broadcast import _expr_child

__all__ = '''
TableExpr TableSymbol Projection Selection Broadcast Join
Reduction join any all sum
min max mean var std count nunique By by Sort Distinct distinct Head head Label
ReLabel relabel Map Apply common_subexpression merge Merge Union selection
projection union broadcast Summary summary'''.split()


class TableExpr(Expr):
    """ Super class for all Table Expressions

    This is not intended to be constructed by users.

    See Also
    --------

    blaze.expr.table.TableSymbol
    """
    __inputs__ = '_child',

    @property
    def dshape(self):
        return datashape.var * self.schema

    @property
    def columns(self):
        return self.fields


class TableSymbol(TableExpr, Symbol):
    """ A Symbol for Tabular data

    This is a leaf in the expression tree

    Examples
    --------

    >>> accounts = TableSymbol('accounts',
    ...                        '{name: string, amount: int, id: int}')
    >>> accounts['amount'] + 1
    accounts['amount'] + 1

    We define a TableSymbol with a name like ``accounts`` and the datashape of
    a single row, called a schema.
    """
    __slots__ = '_name', 'dshape'
    __inputs__ = ()

    def __init__(self, name, dshape=None):
        self._name = name
        if isinstance(dshape, _strtypes):
            dshape = datashape.dshape(dshape)
        if not isdimension(dshape[0]):
            dshape = datashape.var * dshape
        self.dshape = dshape


def unpack(l):
    if isinstance(l, (tuple, list, set)) and len(l) == 1:
        return next(iter(l))
    else:
        return l


class Join(TableExpr):
    """ Join two tables on common columns

    Parameters
    ----------
    lhs : TableExpr
    rhs : TableExpr
    on_left : string
    on_right : string

    Examples
    --------

    >>> names = TableSymbol('names', '{name: string, id: int}')
    >>> amounts = TableSymbol('amounts', '{amount: int, id: int}')

    Join tables based on shared column name
    >>> joined = join(names, amounts, 'id')

    Join based on different column names
    >>> amounts = TableSymbol('amounts', '{amount: int, acctNumber: int}')
    >>> joined = join(names, amounts, 'id', 'acctNumber')

    See Also
    --------

    blaze.expr.table.Merge
    blaze.expr.table.Union
    """
    __slots__ = 'lhs', 'rhs', '_on_left', '_on_right', 'how'
    __inputs__ = 'lhs', 'rhs'

    @property
    def on_left(self):
        if isinstance(self._on_left, tuple):
            return list(self._on_left)
        else:
            return self._on_left

    @property
    def on_right(self):
        if isinstance(self._on_right, tuple):
            return list(self._on_right)
        else:
            return self._on_right

    @property
    def schema(self):
        """

        Examples
        --------

        >>> t = TableSymbol('t', '{name: string, amount: int}')
        >>> s = TableSymbol('t', '{name: string, id: int}')

        >>> join(t, s).schema
        dshape("{ name : string, amount : int32, id : int32 }")

        >>> join(t, s, how='left').schema
        dshape("{ name : string, amount : int32, id : ?int32 }")
        """
        option = lambda dt: dt if isinstance(dt, Option) else Option(dt)

        joined = [[name, dt] for name, dt in self.lhs.schema[0].parameters[0]
                        if name in self.on_left]

        left = [[name, dt] for name, dt in self.lhs.schema[0].parameters[0]
                           if name not in self.on_left]

        right = [[name, dt] for name, dt in self.rhs.schema[0].parameters[0]
                            if name not in self.on_right]

        if self.how in ('right', 'outer'):
            left = [[name, option(dt)] for name, dt in left]
        if self.how in ('left', 'outer'):
            right = [[name, option(dt)] for name, dt in right]

        return dshape(Record(joined + left + right))


def join(lhs, rhs, on_left=None, on_right=None, how='inner'):
    if not on_left and not on_right:
        on_left = on_right = unpack(list(sorted(
            set(lhs.fields) & set(rhs.fields),
            key=lhs.fields.index)))
    if not on_right:
        on_right = on_left
    if isinstance(on_left, tuple):
        on_left = list(on_left)
    if isinstance(on_right, tuple):
        on_right = list(on_right)
    if get(on_left, lhs.schema[0]) != get(on_right, rhs.schema[0]):
        raise TypeError("Schema's of joining columns do not match")
    _on_left = tuple(on_left) if isinstance(on_left, list) else on_left
    _on_right = (tuple(on_right) if isinstance(on_right, list)
                        else on_right)

    how = how.lower()
    if how not in ('inner', 'outer', 'left', 'right'):
        raise ValueError("How parameter should be one of "
                         "\n\tinner, outer, left, right."
                         "\nGot: %s" % how)

    return Join(lhs, rhs, _on_left, _on_right, how)


join.__doc__ = Join.__doc__

def _names_and_types(expr):
    schema = expr.dshape.measure
    if isinstance(schema, Option):
        schema = schema.ty
    if isinstance(schema, Record):
        return schema.names, schema.types
    if isinstance(schema, Unit):
        return [expr._name], [expr.dshape.measure]
    raise ValueError("Unable to determine name and type of %s" % expr)


class By(TableExpr):
    """ Split-Apply-Combine Operator

    Examples
    --------

    >>> t = TableSymbol('t', '{name: string, amount: int, id: int}')
    >>> e = by(t['name'], t['amount'].sum())

    >>> data = [['Alice', 100, 1],
    ...         ['Bob', 200, 2],
    ...         ['Alice', 50, 3]]

    >>> from blaze.compute.python import compute
    >>> sorted(compute(e, data))
    [('Alice', 150), ('Bob', 200)]
    """

    __slots__ = 'grouper', 'apply'

    @property
    def _child(self):
        return common_subexpression(self.grouper, self.apply)

    @property
    def schema(self):
        grouper_names, grouper_types = _names_and_types(self.grouper)
        apply_names, apply_types = _names_and_types(self.apply)

        names = grouper_names + apply_names
        types = grouper_types + apply_types

        return dshape(Record(list(zip(names, types))))


@dispatch(Expr, (Summary, Reduction))
def by(grouper, apply):
    return By(grouper, apply)


@dispatch(Expr)
def by(grouper, **kwargs):
    return By(grouper, summary(**kwargs))


def count_values(expr, sort=True):
    """
    Count occurrences of elements in this column

    Sort by counts by default
    Add ``sort=False`` keyword to avoid this behavior.
    """
    result = by(expr, count=expr.count())
    if sort:
        result = result.sort('count', ascending=False)
    return result

class Apply(TableExpr):
    """ Apply an arbitrary Python function onto a Table

    Examples
    --------

    >>> t = TableSymbol('t', '{name: string, amount: int}')
    >>> h = Apply(hash, t)  # Hash value of resultant table

    Optionally provide extra datashape information

    >>> h = Apply(hash, t, dshape='real')

    Apply brings a function within the expression tree.
    The following transformation is often valid

    Before ``compute(Apply(f, expr), ...)``
    After  ``f(compute(expr, ...)``

    See Also
    --------

    blaze.expr.table.Map
    """
    __slots__ = '_child', 'func', '_dshape'

    def __init__(self, func, child, dshape=None):
        self._child = child
        self.func = func
        self._dshape = dshape

    @property
    def schema(self):
        if iscollection(self.dshape):
            return self.dshape.subshape[0]
        else:
            raise TypeError("Non-tabular datashape, %s" % self.dshape)

    @property
    def dshape(self):
        if self._dshape:
            return dshape(self._dshape)
        else:
            raise NotImplementedError("Datashape of arbitrary Apply not defined")


from datashape.predicates import (iscollection, isscalar, isrecord, isboolean,
        isnumeric)
from datashape import Unit, Record, to_numpy_dtype, bool_
from .expressions import schema_method_list, dshape_method_list
from .broadcast import isnan

schema_method_list.extend([
    (isnumeric, set([isnan])),
    (isscalar,  set([label, relabel])),
    (isrecord,  set([relabel])),
    ])

dshape_method_list.extend([
    (iscollection, set([count_values])),
    ])
