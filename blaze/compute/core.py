from __future__ import absolute_import, division, print_function

import numbers
import ast
import operator as op
from datetime import date, datetime

from toolz import first, merge, keyfilter, complement

from ..compatibility import basestring
from ..expr.core import Expr
from ..expr import TableSymbol, Union, eval_str
from ..dispatch import dispatch
import math

__all__ = ['compute', 'compute_one', 'drop', 'create_index']

base = (numbers.Real, str, date, datetime)


@dispatch(object, object)
def compute_one(a, b, **kwargs):
    raise NotImplementedError("Blaze does not know how to compute "
                              "expression of type `%s` on data of type `%s`"
                              % (type(a).__name__, type(b).__name__))


@dispatch(base)
def compute_one(a, **kwargs):
    return a


@dispatch((list, tuple))
def compute_one(seq, scope={}, **kwargs):
    return type(seq)(compute(item, scope, **kwargs) for item in seq)


@dispatch(Expr, object)
def compute(expr, o, **kwargs):
    """ Compute against single input

    Assumes that only one TableSymbol exists in expression

    >>> t = TableSymbol('t', '{name: string, balance: int}')
    >>> deadbeats = t[t['balance'] < 0]['name']

    >>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
    >>> # list(compute(deadbeats, {t: data}))
    >>> list(compute(deadbeats, data))
    ['Bob', 'Charlie']
    """
    ts = set([x for x in expr.subterms() if isinstance(x, TableSymbol)])
    if len(ts) == 1:
        return compute(expr, {first(ts): o}, **kwargs)
    else:
        raise ValueError("Give compute dictionary input, got %s" % str(o))


@dispatch(object)
def compute_down(expr):
    """ Compute the expression on the entire inputs

    inputs match up to leaves of the expression
    """
    return expr


def top_to_bottom(d, expr):
    """ Processes an expression top-down then bottom-up """
    # Base case: expression is in dict, return associated data
    if expr in d:
        return d[expr]

    # See if we have a direct computation path
    if (hasattr(expr, 'leaves') and compute_down.resolve(
            (type(expr),) + tuple(type(d.get(leaf)) for leaf in expr.leaves()))):
        leaves = [d[leaf] for leaf in expr.leaves()]
        return compute_down(expr, *leaves)
    else:
        # Compute children of this expression
        children = ([top_to_bottom(d, child) for child in expr.inputs]
                    if hasattr(expr, 'inputs') else [])

        # Compute this expression given the children
        return compute_one(expr, *children, scope=d)


def bottom_up(d, expr):
    """
    Process an expression from the leaves upwards

    Parameters
    ----------

    d : dict mapping {TableSymbol: data}
        Maps expressions to data elements, likely at the leaves of the tree
    expr : Expr
        Expression to compute

    Helper function for ``compute``
    """
    # Base case: expression is in dict, return associated data
    if expr in d:
        return d[expr]

    # Compute children of this expression
    children = ([bottom_up(d, child) for child in expr.inputs]
                if hasattr(expr, 'inputs') else [])

    # Compute this expression given the children
    result = compute_one(expr, *children, scope=d)

    return result


@dispatch(Expr, dict)
def pre_compute(expr, d):
    """ Transform expr prior to calling ``compute`` """
    return expr


@dispatch(Expr, object, dict)
def post_compute(expr, result, d):
    """ Effects after the computation is complete """
    return result


@dispatch(Expr, dict)
def compute(expr, d):
    """ Compute expression against data sources

    >>> t = TableSymbol('t', '{name: string, balance: int}')
    >>> deadbeats = t[t['balance'] < 0]['name']

    >>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
    >>> list(compute(deadbeats, {t: data}))
    ['Bob', 'Charlie']
    """
    expr = pre_compute(expr, d)
    result = top_to_bottom(d, expr)
    return post_compute(expr, result, d)


@dispatch(Union, (list, tuple))
def compute_one(t, children, **kwargs):
    return compute_one(t, children[0], tuple(children))


@dispatch(object, basestring, (basestring, list, tuple))
def create_index(t, index_name, column_name_or_names):
    """Create an index on a column.

    Parameters
    ----------
    o : table-like
    index_name : str
        The name of the index to create
    column_name_or_names : basestring, list, tuple
        A column name to index on, or a list or tuple for a composite index

    Examples
    --------
    >>> # Using SQLite
    >>> from blaze import SQL
    >>> # create a table called 'tb', in memory
    >>> sql = SQL('sqlite:///:memory:', 'tb',
    ...           schema='{id: int64, value: float64, categ: string}')
    >>> data = [(1, 2.0, 'a'), (2, 3.0, 'b'), (3, 4.0, 'c')]
    >>> sql.extend(data)
    >>> # create an index on the 'id' column (for SQL we must provide a name)
    >>> sql.table.indexes
    set()
    >>> create_index(sql, 'id', name='id_index')
    >>> sql.table.indexes
    {Index('id_index', Column('id', BigInteger(), table=<tb>, nullable=False))}
    """
    raise NotImplementedError("create_index not implemented for type %r" %
                              type(t).__name__)


@dispatch(object)
def drop(rsrc):
    """Remove a resource.

    Parameters
    ----------
    rsrc : CSV, SQL, tables.Table, pymongo.Collection
        A resource that will be removed. For example, calling ``drop(csv)`` will
        delete the CSV file.

    Examples
    --------
    >>> # Using SQLite
    >>> from blaze import SQL, into
    >>> # create a table called 'tb', in memory
    >>> sql = SQL('sqlite:///:memory:', 'tb',
    ...           schema='{id: int64, value: float64, categ: string}')
    >>> data = [(1, 2.0, 'a'), (2, 3.0, 'b'), (3, 4.0, 'c')]
    >>> sql.extend(data)
    >>> into(list, sql)
    [(1, 2.0, 'a'), (2, 3.0, 'b'), (3, 4.0, 'c')]
    >>> sql.table.exists(sql.engine)
    True
    >>> drop(sql)
    >>> sql.table.exists(sql.engine)
    False
    """
    raise NotImplementedError("drop not implemented for type %r" %
                              type(rsrc).__name__)
