"""SparkSQL backend for blaze.

Notes
-----
Translation happens via the Hive sqlalchemy dialect, which is then sent to
SparkSQL.
"""

from __future__ import absolute_import, division, print_function

from datashape.predicates import isrecord
import sqlalchemy as sa
from toolz import pipe, curry
from toolz.curried import filter, map
from odo.backends.sql import dshape_to_alchemy

from ..dispatch import dispatch
from ..expr import Expr, Field, symbol
from .core import compute
from .utils import literalquery
from .spark import Dummy

__all__ = []


try:
    from pyspark import SQLContext
    from pyhive.sqlalchemy_hive import HiveDialect
except ImportError:
    SQLContext = Dummy


def make_sqlalchemy_table(expr):
    name = expr._name
    columns = dshape_to_alchemy(expr.dshape)
    return sa.Table(name, sa.MetaData(), *columns)


@curry
def istable(db, t):
    return (isinstance(t, Field) and isrecord(t.dshape.measure) and
            t._child.isidentical(db))


@dispatch(Expr, SQLContext)
def compute_down(expr, data, **kwargs):
    """ Compile a blaze expression to a sparksql expression"""
    leaves = expr._leaves()

    # make sure we only have a single leaf node
    if len(leaves) != 1:
        raise ValueError('Must compile from exactly one root database')

    leaf, = leaves

    # field expressions on the database are Field instances with a record
    # measure whose immediate child is the database leaf
    tables = pipe(expr._subterms(), filter(istable(leaf)), list)

    # raise if we don't have tables in our database
    if not tables:
        raise ValueError('Expressions not referencing a table cannot be '
                         'compiled')

    # make new symbols for each table
    new_leaves = [symbol(t._name, t.dshape) for t in tables]

    # sub them in the expression
    expr = expr._subs(dict(zip(tables, new_leaves)))

    # compute using sqlalchemy
    scope = dict(zip(new_leaves, map(make_sqlalchemy_table, tables)))
    query = compute(expr, scope)

    # interpolate params
    compiled = literalquery(query, dialect=HiveDialect())
    return data.sql(str(compiled))
