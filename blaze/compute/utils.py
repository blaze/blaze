from __future__ import absolute_import, division, print_function

from datetime import datetime
from decimal import Decimal

import sqlalchemy as sa
import sqlalchemy.orm
from toolz import curry
from datashape.predicates import isrecord
from ..expr import Field
from odo.backends.sql import dshape_to_alchemy

# This was taken from the following StackOverflow post
# http://stackoverflow.com/questions/5631078/sqlalchemy-print-the-actual-query
# answer by bukzor http://stackoverflow.com/users/146821/bukzor


def literalquery(statement, dialect=None):
    """Generate an SQL expression string with bound parameters rendered inline
    for the given SQLAlchemy statement.

    WARNING: This method of escaping is insecure, incomplete, and for debugging
    purposes only. Executing SQL statements with inline-rendered user values is
    extremely insecure.
    """
    if isinstance(statement, sqlalchemy.orm.Query):
        if dialect is None:
            dialect = statement.session.get_bind(
                statement._mapper_zero_or_none()
            ).dialect
        statement = statement.statement
    if dialect is None:
        dialect = getattr(statement.bind, 'dialect', None)
    if dialect is None:
        from sqlalchemy.dialects import mysql
        dialect = mysql.dialect()

    Compiler = type(statement._compiler(dialect))

    class LiteralCompiler(Compiler):
        visit_bindparam = Compiler.render_literal_bindparam

        def render_literal_value(self, value, type_):
            if isinstance(value, (Decimal, long)):
                return str(value)
            elif isinstance(value, datetime):
                return repr(str(value))
            else:  # fallback
                value = super(LiteralCompiler, self).render_literal_value(
                    value, type_,
                )
                if isinstance(value, unicode):
                    return value.encode('UTF-8')
                else:
                    return value

    return LiteralCompiler(dialect, statement)


def make_sqlalchemy_table(expr):
    return sa.Table(expr._name, sa.MetaData(), *dshape_to_alchemy(expr.dshape))


@curry
def istable(db, t):
    return (isinstance(t, Field) and isrecord(t.dshape.measure) and
            t._child.isidentical(db))
