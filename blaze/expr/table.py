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

class Reduction(Expr):
    """ A column-wise reduction

    Blaze supports the same class of reductions as NumPy and Pandas.

        sum, min, max, any, all, mean, var, std, count, nunique

    Examples
    --------

    >>> t = TableSymbol('t', '{name: string, amount: int, id: int}')
    >>> e = t['amount'].sum()

    >>> data = [['Alice', 100, 1],
    ...         ['Bob', 200, 2],
    ...         ['Alice', 50, 3]]

    >>> from blaze.compute.python import compute
    >>> compute(e, data)
    350
    """
    __slots__ = '_child',
    _dtype = None

    @property
    def dshape(self):
        return dshape(self._dtype)

    @property
    def symbol(self):
        return type(self).__name__

    @property
    def _name(self):
        try:
            return self._child._name + '_' + type(self).__name__
        except (AttributeError, ValueError, TypeError):
            return type(self).__name__



class any(Reduction):
    _dtype = ct.bool_

class all(Reduction):
    _dtype = ct.bool_

class sum(Reduction):
    @property
    def _dtype(self):
        schema = self._child.schema[0]
        if isinstance(schema, Record) and len(schema.types) == 1:
            return first(schema.types)
        else:
            return schema

class max(Reduction):
    @property
    def _dtype(self):
        schema = self._child.schema[0]
        if isinstance(schema, Record) and len(schema.types) == 1:
            return first(schema.types)
        else:
            return schema

class min(Reduction):
    @property
    def _dtype(self):
        schema = self._child.schema[0]
        if isinstance(schema, Record) and len(schema.types) == 1:
            return first(schema.types)
        else:
            return schema

class mean(Reduction):
    _dtype = ct.real

class var(Reduction):
    """Variance

    Parameters
    ----------
    child : Expr
        An expression
    unbiased : bool, optional
        Compute an unbiased estimate of the population variance if this is
        ``True``. In NumPy and pandas, this parameter is called ``ddof`` (delta
        degrees of freedom) and is equal to 1 for unbiased and 0 for biased.
    """
    __slots__ = '_child', 'unbiased'

    _dtype = ct.real

    def __init__(self, child, unbiased=False):
        super(var, self).__init__(child, unbiased)

class std(Reduction):
    """Standard Deviation

    Parameters
    ----------
    child : Expr
        An expression
    unbiased : bool, optional
        Compute the square root of an unbiased estimate of the population
        variance if this is ``True``.

        .. warning::

            This does *not* return an unbiased estimate of the population
            standard deviation.

    See Also
    --------
    var
    """
    __slots__ = '_child', 'unbiased'

    _dtype = ct.real

    def __init__(self, child, unbiased=False):
        super(std, self).__init__(child, unbiased)

class count(Reduction):
    _dtype = ct.int_

class nunique(Reduction):
    _dtype = ct.int_


class Summary(Expr):
    """ A collection of named reductions

    Examples
    --------

    >>> t = TableSymbol('t', '{name: string, amount: int, id: int}')
    >>> expr = summary(number=t.id.nunique(), sum=t.amount.sum())

    >>> data = [['Alice', 100, 1],
    ...         ['Bob', 200, 2],
    ...         ['Alice', 50, 1]]

    >>> from blaze.compute.python import compute
    >>> compute(expr, data)
    (2, 350)
    """
    __slots__ = '_child', 'names', 'values'

    @property
    def dshape(self):
        return dshape(Record(list(zip(self.names,
                                      [v._dtype for v in self.values]))))

    def __str__(self):
        return 'summary(' + ', '.join('%s=%s' % (name, str(val))
                for name, val in zip(self.fields, self.values)) + ')'


def summary(**kwargs):
    items = sorted(kwargs.items(), key=first)
    names = tuple(map(first, items))
    values = tuple(map(toolz.second, items))
    child = common_subexpression(*values)

    if len(kwargs) == 1 and not iscollection(child.dshape):
        while not iscollection(child.dshape):
            children = [i for i in child._inputs if isinstance(i, Expr)]
            if len(children) == 1:
                child = children[0]
            else:
                raise ValueError()

    return Summary(child, names, values)


summary.__doc__ = Summary.__doc__


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

class ReLabel(ElemWise):
    """
    Table with same content but with new labels

    Examples
    --------

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
    >>> accounts.schema
    dshape("{ name : string, amount : int32 }")
    >>> accounts.relabel({'amount': 'balance'}).schema
    dshape("{ name : string, balance : int32 }")

    See Also
    --------

    blaze.expr.table.Label
    """
    __slots__ = '_child', 'labels'

    @property
    def schema(self):
        subs = dict(self.labels)
        d = self._child.dshape.measure.dict

        return DataShape(Record([[subs.get(name, name), dtype]
            for name, dtype in self._child.dshape.measure.parameters[0]]))


def relabel(child, labels):
    if isinstance(labels, dict):  # Turn dict into tuples
        labels = tuple(sorted(labels.items()))
    if isscalar(child.dshape.measure):
        if child._name == labels[0][0]:
            return child.label(labels[0][1])
        else:
            return child
    return ReLabel(child, labels)

relabel.__doc__ = ReLabel.__doc__


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
    (isboolean, set([any, all])),
    (isnumeric, set([mean, isnan, sum, mean, min, max, std, var])),
    (isscalar,  set([label, relabel])),
    (isrecord,  set([relabel])),
    ])

dshape_method_list.extend([
    (iscollection, set([count, nunique, count_values])),
    ])
