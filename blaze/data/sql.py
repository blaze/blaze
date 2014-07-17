from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from datetime import date, datetime, time
from decimal import Decimal
from dynd import nd
import sqlalchemy as sql
import datashape
from datashape import dshape, var, Record, Option, isdimension
from itertools import chain
from toolz import first

from ..dispatch import dispatch
from ..utils import partition_all
from ..compatibility import basestring
from .core import DataDescriptor
from .utils import coerce_row_to_dict
from ..compatibility import _inttypes, _strtypes

# http://docs.sqlalchemy.org/en/latest/core/types.html

types = {'int64': sql.types.BigInteger,
         'int32': sql.types.Integer,
         'int': sql.types.Integer,
         'int16': sql.types.SmallInteger,
         'float': sql.types.Float,
         'float32': sql.types.REAL,
         'float64': sql.types.Float,
         'string': sql.types.String,  # Probably just use only this
         'date': sql.types.Date,
         'time': sql.types.Time,
         'datetime': sql.types.DateTime,
         'bool': sql.types.Boolean,
#         ??: sql.types.LargeBinary,
#         Decimal: sql.types.Numeric,
#         ??: sql.types.PickleType,
#         unicode: sql.types.Unicode,
#         unicode: sql.types.UnicodeText,
#         str: sql.types.Text,  # ??
         }


revtypes = dict(map(reversed, types.items()))

revtypes.update({sql.types.VARCHAR: 'string',
                 sql.types.DATETIME: 'datetime',
                 sql.types.TIMESTAMP: 'datetime',
                 sql.types.FLOAT: 'float64',
                 sql.types.DATE: 'date',
                 sql.types.BIGINT: 'int64',
                 sql.types.INTEGER: 'int',
                 sql.types.Float: 'float64'})


@dispatch(sql.sql.type_api.TypeEngine)
def discover(typ):
    if type(typ) in revtypes:
        return dshape(revtypes[type(typ)])[0]
    else:
        for k, v in revtypes.items():
            if isinstance(typ, k):
                return v
    raise NotImplementedError("No SQL-datashape match for type %s" % typ)


@dispatch(sql.Column)
def discover(col):
    if col.nullable:
        return Record([[col.name, Option(discover(col.type))]])
    else:
        return Record([[col.name, discover(col.type)]])


@dispatch(sql.Table)
def discover(t):
    return var * Record(list(sum([discover(c).parameters[0] for c in t.columns], ())))


@dispatch(sql.engine.base.Engine, str)
def discover(engine, tablename):
    metadata = sql.MetaData()
    metadata.reflect(engine)
    table = metadata.tables[tablename]
    return discover(table)


def dshape_to_alchemy(dshape):
    """

    >>> dshape_to_alchemy('int')
    <class 'sqlalchemy.sql.sqltypes.Integer'>

    >>> dshape_to_alchemy('string')
    <class 'sqlalchemy.sql.sqltypes.String'>

    >>> dshape_to_alchemy('{name: string, amount: int}')
    [Column('name', String(), table=None, nullable=False), Column('amount', Integer(), table=None, nullable=False)]

    >>> dshape_to_alchemy('{name: ?string, amount: ?int}')
    [Column('name', String(), table=None), Column('amount', Integer(), table=None)]
    """
    if isinstance(dshape, _strtypes):
        dshape = datashape.dshape(dshape)
    if isinstance(dshape, Option):
        return dshape_to_alchemy(dshape.ty)
    if str(dshape) in types:
        return types[str(dshape)]
    if isinstance(dshape, datashape.Record):
        return [sql.Column(name,
                           dshape_to_alchemy(typ),
                           nullable=isinstance(typ[0], Option))
                    for name, typ in dshape.parameters[0]]
    if isinstance(dshape, datashape.DataShape):
        if isdimension(dshape[0]):
            return dshape_to_alchemy(dshape[1])
        else:
            return dshape_to_alchemy(dshape[0])
    raise NotImplementedError("No SQLAlchemy dtype match for datashape: %s"
            % dshape)


class SQL(DataDescriptor):
    """
    A Blaze data descriptor to expose a SQL database.

    >>> dd = SQL('sqlite:///:memory:', 'accounts',
    ...          schema='{name: string, amount: int}')

    Insert into database

    >>> dd.extend([('Alice', 100), ('Bob', 200)])

    Select all from table
    >>> list(dd) # doctest: +SKIP
    [('Alice', 100), ('Bob', 200)]

    Verify that we're actually touching the database

    >>> with dd.engine.connect() as conn: # doctest: +SKIP
    ...     print(list(conn.execute('SELECT * FROM accounts')))
    [('Alice', 100), ('Bob', 200)]


    Parameters
    ----------
    engine : string, A SQLAlchemy engine
        uri of database
        or SQLAlchemy engine
    table : string
        The name of the table
    schema : string, list of Columns
        The datashape/schema of the database
        Possibly a list of SQLAlchemy columns
    """
    immutable = False
    deferred = False
    appendable = True

    @property
    def remote(self):
        return self.engine.dialect.name != 'sqlite'

    @property
    def persistent(self):
        return self.engine.url != 'sqlite:///:memory:'

    def __init__(self, engine, tablename, primary_key='', schema=None):
        if isinstance(engine, _strtypes):
            engine = sql.create_engine(engine)
        self.engine = engine
        self.tablename = tablename
        metadata = sql.MetaData()

        if engine.has_table(tablename):
            metadata.reflect(engine)
            table = metadata.tables[tablename]
            engine_schema = discover(table).subshape[0]
            # if schema and dshape(schema) != engine_schema:
                # raise ValueError("Mismatched schemas:\n"
                #                 "\tIn database: %s\n"
                #                 "\nGiven: %s" % (engine_schema, schema))
            if not schema:
                schema = engine_schema
        elif isinstance(schema, (_strtypes, datashape.DataShape)):
            columns = dshape_to_alchemy(schema)
            for column in columns:
                if column.name == primary_key:
                    column.primary_key = True
            table = sql.Table(tablename, metadata, *columns)
        else:
            raise ValueError('Must provide schema or point to valid table. '
                             'Table %s does not exist' % tablename)

        self._schema = datashape.dshape(schema)
        self.table = table
        metadata.create_all(engine)

    def _iter(self):
        with self.engine.connect() as conn:
            result = conn.execute(sql.sql.select([self.table]))
            for item in result:
                yield item

    @property
    def dshape(self):
        return datashape.Var() * self.schema

    def extend(self, rows):
        rows = iter(rows)
        try:
            row = next(rows)
        except StopIteration:
            return
        rows = chain([row], rows)
        # Coerce rows to dicts
        if isinstance(row, (tuple, list)):
            names = self.schema[0].names
            rows = (dict(zip(names, row)) for row in rows)
        with self.engine.connect() as conn:
            for chunk in partition_all(1000, rows):  # TODO: 1000 is hardcoded
                conn.execute(self.table.insert(), chunk)

    def chunks(self, blen=1000):
        for chunk in partition_all(blen, iter(self)):
            dshape = str(len(chunk)) + ' * ' + str(self.schema)
            yield nd.array(chunk, dtype=dshape)

    def _query(self, query, transform=lambda x: x):
        with self.engine.connect() as conn:
            result = conn.execute(query)
            for item in result:
                yield transform(item)

    def _get_py(self, key):
        if not isinstance(key, tuple):
            key = (key, slice(0, None))
        if ((len(key) != 2 and not isinstance(key[0], (_inttypes, slice, _strtypes)))
            or (isinstance(key[0], _inttypes) and key[0] != 0)):
            raise ValueError("Limited indexing supported for SQL")
        rows, cols = key
        transform = lambda x: x
        single_item = False
        if rows == 0:
            single_item = True
            rows = slice(0, 1)
        if (rows.start not in (0, None) or rows.step not in (1, None)):
            raise ValueError("Limited indexing supported for SQL")
        if isinstance(cols, slice):
            cols = self.schema[0].names[cols]
        if isinstance(cols, _strtypes):
            transform = lambda x: x[0]
            columns = [getattr(self.table.c, cols)]
        if isinstance(cols, _inttypes):
            transform = lambda x: x[0]
            columns = [getattr(self.table.c, self.schema[0].names[cols])]
        else:
            columns = [getattr(self.table.c, x) if isinstance(x, _strtypes)
                       else getattr(self.table.c, self.schema[0].names[x])
                       for x in cols]

        query = sql.sql.select(columns).limit(rows.stop)
        result = self._query(query, transform)

        if single_item:
            return next(result)
        else:
            return result
