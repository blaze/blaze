"""SparkSQL backend for blaze.

Notes
-----
Translation happens via the Hive sqlalchemy dialect, which is then sent to
SparkSQL.
"""

from __future__ import absolute_import, division, print_function

from operator import and_

from toolz import pipe
from toolz.curried import filter, map

from ..dispatch import dispatch
from ..expr import Expr, symbol, Join
from .core import compute
from .utils import literalquery, istable, make_sqlalchemy_table
from ..utils import listpack
from .spark import Dummy, jgetattr

__all__ = []


try:
    from pyspark import SQLContext
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyhive.sqlalchemy_hive import HiveDialect
except ImportError:
    SQLContext = Dummy


join_types = {
    'left': 'left_outer',
    'right': 'right_outer'
}


@dispatch(Join, SparkDataFrame, SparkDataFrame)
def compute_up(t, lhs, rhs, **kwargs):
    ands = [getattr(lhs, left) == getattr(rhs, right)
            for left, right in zip(*map(listpack, (t.on_left, t.on_right)))]

    joined = lhs.join(rhs, reduce(and_, ands), join_types.get(t.how, t.how))

    prec, sec = (rhs, lhs) if t.how == 'right' else (lhs, rhs)
    cols = [jgetattr(prec, f, jgetattr(sec, f, None)) for f in t.fields]
    assert all(c is not None for c in cols)
    return joined.select(*cols)


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
