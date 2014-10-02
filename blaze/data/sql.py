from __future__ import absolute_import, division, print_function


import sys
import os
import warnings
import subprocess

from dynd import nd
import sqlalchemy as sa
import sqlalchemy
import datashape
from datashape import dshape, var, Record, Option, isdimension
from itertools import chain
from toolz import merge

from ..data.utils import raw
from ..dispatch import dispatch
from ..utils import partition_all
from .core import DataDescriptor
from ..compatibility import _inttypes, _strtypes
from .csv import CSV

# http://docs.sqlalchemy.org/en/latest/core/types.html

types = {'int64': sa.types.BigInteger,
         'int32': sa.types.Integer,
         'int': sa.types.Integer,
         'int16': sa.types.SmallInteger,
         'float32': sa.types.Float(precision=24),  # sqlalchemy uses mantissa
         'float64': sa.types.Float(precision=53),  # for precision
         'float': sa.types.Float(precision=53),
         'real': sa.types.Float(precision=53),
         'string': sa.types.Text,
         'date': sa.types.Date,
         'time': sa.types.Time,
         'datetime': sa.types.DateTime,
         'bool': sa.types.Boolean,
         #         ??: sa.types.LargeBinary,
         #         Decimal: sa.types.Numeric,
         #         ??: sa.types.PickleType,
         #         unicode: sa.types.Unicode,
         #         unicode: sa.types.UnicodeText,
         # str: sa.types.Text,  # ??
         }

revtypes = dict(map(reversed, types.items()))

revtypes.update({sa.types.VARCHAR: 'string',
                 sa.types.String: 'string',
                 sa.types.Unicode: 'string',
                 sa.types.DATETIME: 'datetime',
                 sa.types.TIMESTAMP: 'datetime',
                 sa.types.FLOAT: 'float64',
                 sa.types.DATE: 'date',
                 sa.types.BIGINT: 'int64',
                 sa.types.INTEGER: 'int',
                 sa.types.Float: 'float64'})


@dispatch(sa.sql.type_api.TypeEngine)
def discover(typ):
    if typ in revtypes:
        return dshape(revtypes[typ])[0]
    if type(typ) in revtypes:
        return dshape(revtypes[type(typ)])[0]
    else:
        for k, v in revtypes.items():
            if isinstance(k, type) and isinstance(typ, k):
                return v
            if k == typ:
                return v
    raise NotImplementedError("No SQL-datashape match for type %s" % typ)


@dispatch(sa.Column)
def discover(col):
    if col.nullable:
        return Record([[col.name, Option(discover(col.type))]])
    else:
        return Record([[col.name, discover(col.type)]])


@dispatch(sa.Table)
def discover(t):
    return var * Record(list(sum([discover(c).parameters[0]
                                  for c in t.columns], ())))


@dispatch(sa.engine.base.Engine, _strtypes)
def discover(engine, tablename):
    metadata = sa.MetaData()
    metadata.reflect(engine)
    table = metadata.tables[tablename]
    return discover(table)


def dshape_to_alchemy(dshape):
    """

    >>> dshape_to_alchemy('int')
    <class 'sqlalchemy.sql.sqltypes.Integer'>

    >>> dshape_to_alchemy('string')
    <class 'sqlalchemy.sql.sqltypes.Text'>

    >>> dshape_to_alchemy('{name: string, amount: int}')
    [Column('name', Text(), table=None, nullable=False), Column('amount', Integer(), table=None, nullable=False)]

    >>> dshape_to_alchemy('{name: ?string, amount: ?int}')
    [Column('name', Text(), table=None), Column('amount', Integer(), table=None)]
    """
    if isinstance(dshape, _strtypes):
        dshape = datashape.dshape(dshape)
    if isinstance(dshape, Option):
        return dshape_to_alchemy(dshape.ty)
    if str(dshape) in types:
        return types[str(dshape)]
    if isinstance(dshape, datashape.Record):
        return [sa.Column(name,
                          dshape_to_alchemy(typ),
                          nullable=isinstance(typ[0], Option))
                for name, typ in dshape.parameters[0]]
    if isinstance(dshape, datashape.DataShape):
        if isdimension(dshape[0]):
            return dshape_to_alchemy(dshape[1])
        else:
            return dshape_to_alchemy(dshape[0])
    if isinstance(dshape, datashape.String):
        if dshape[0].fixlen is None:
            return sa.types.Text
        if 'U' in dshape.encoding:
            return sa.types.Unicode(length=dshape[0].fixlen)
        if 'A' in dshape.encoding:
            return sa.types.String(length=dshape[0].fixlen)
    if isinstance(dshape, datashape.DateTime):
        if dshape.tz:
            return sa.types.DateTime(timezone=True)
        else:
            return sa.types.DateTime(timezone=False)
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
    def __init__(self, engine, tablename, primary_key='', schema=None):
        if isinstance(engine, _strtypes):
            engine = sa.create_engine(engine)
        self.engine = engine
        self.tablename = tablename
        metadata = sa.MetaData()

        if engine.has_table(tablename):
            metadata.reflect(engine)
            table = metadata.tables[tablename]
            engine_schema = discover(table).subshape[0]

            if not schema:
                schema = engine_schema

        elif isinstance(schema, (_strtypes, datashape.DataShape)):
            columns = dshape_to_alchemy(schema)
            for column in columns:
                if column.name == primary_key:
                    column.primary_key = True
            table = sa.Table(tablename, metadata, *columns)
        else:
            raise ValueError('Must provide schema or point to valid table. '
                             'Table %s does not exist' % tablename)

        self._schema = datashape.dshape(schema)
        self.table = table
        metadata.create_all(engine)

    def _iter(self):
        with self.engine.connect() as conn:
            result = conn.execute(sa.sql.select([self.table]))
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

        query = sa.sql.select(columns).limit(rows.stop)
        result = self._query(query, transform)

        if single_item:
            return next(result)
        else:
            return result


@dispatch(SQL, CSV)
def into(sql, csv, if_exists="replace", **kwargs):
    """
    Parameters
    ----------
    if_exists : str, optional, {replace, append, fail}
        The behavior defining what happens when a table exists.
    format_str : string, optional, {text, csv}
         The data format.
    delimiter : string
        Delimiter character
    header : bool, optional
        Flag to define file contains a header (only to be used with CSV)
    quotechar : str, optional
        Single byte quote. For example, "1997","Ford","E350".
    escapechar : str, optional
        A single one-byte character for defining escape characters.
    encoding : str, optional
        An encoding name (POSTGRES): utf8, latin-1, ascii.
    na_value : str, optional
        String to use to indicate NULL values


    ##NOT IMPLEMENTED
    # FORCE_QUOTE { ( column_name [, ...] ) | * }
    # FORCE_NOT_NULL ( column_name [, ...] )
    # OIDS : bool
    #     Specifies copying the OID for each row

    """
    dbtype = getattr(sql, 'dbtype', sql.engine.name)
    db = sql.engine.url.database
    abspath = csv._abspath
    tblname = sql.tablename

    # using dialect mappings to support multiple term mapping for similar
    # concepts
    from .dialect_mappings import dialect_terms

    def retrieve_kwarg(term):
        terms = [k for k, v in dialect_terms.items() if v == term]
        for t in terms:
            val = kwargs.get(t)
            if val:
                return val

    for k in kwargs.keys():
        if k not in dialect_terms:
            raise KeyError('%s not found in dialect mapping' % k)

    format_str = retrieve_kwarg('format_str') or 'csv'
    encoding = retrieve_kwarg('encoding') or 'utf8' if dbtype == 'mysql' else 'latin1'
    delimiter = retrieve_kwarg('delimiter') or csv.dialect['delimiter']
    na_value = retrieve_kwarg('na_value') or ''
    quotechar = retrieve_kwarg('quotechar') or '"'
    escapechar = retrieve_kwarg('escapechar') or quotechar
    header = retrieve_kwarg('header') or csv.header
    lineterminator = retrieve_kwarg('lineterminator') or os.linesep

    skiprows = csv.header or 0  # None or 0 returns 0
    skiprows = retrieve_kwarg('skiprows') or int(skiprows)
    mysql_local = retrieve_kwarg('local') or ''

    copy_info = {'abspath': abspath,
                 'tblname': tblname,
                 'db': db,
                 'format_str': format_str,
                 'delimiter': delimiter,
                 'na_value': na_value,
                 'quotechar': quotechar,
                 'escapechar': escapechar,
                 'lineterminator': raw(lineterminator),
                 'skiprows': skiprows,
                 'header': header,
                 'encoding': encoding,
                 'mysql_local': mysql_local}

    return sql.into(csv, **merge(kwargs, copy_info, dict(if_exists=if_exists)))


class SQLIntoMixin(object):

    def pre_into(self, if_exists='replace', **kwargs):
        engine = self.engine
        tblname = self.tablename

        if if_exists == 'replace':
            if engine.has_table(tblname):

                # drop old table
                metadata = sqlalchemy.MetaData()
                metadata.reflect(engine, only=[tblname])
                t = metadata.tables[tblname]
                t.drop(engine)

                # create a new one
                self.table.create(engine)
        return kwargs

    def into(self, b, **kwargs):
        self.pre_into(**kwargs)
        return self._into(b, **kwargs)


EXTEND_CODES = set([1083,  # field separator argument is not what is expected
                    ])


class MySQL(SQL, SQLIntoMixin):
    dbtype = 'mysql'

    def _into(self, b, **kwargs):
        # no null handling
        sql_stmnt = """
                    LOAD DATA {mysql_local} INFILE '{abspath}'
                    INTO TABLE {tblname}
                    CHARACTER SET {encoding}
                    FIELDS
                        TERMINATED BY '{delimiter}'
                        ENCLOSED BY '{quotechar}'
                        ESCAPED BY '{escapechar}'
                    LINES TERMINATED by '{lineterminator}'
                    IGNORE {skiprows} LINES;"""
        sql_stmnt = sql_stmnt.format(**kwargs)

        try:
            with self.engine.begin() as conn:
                conn.execute(sql_stmnt)
        except sa.exc.InternalError as e:
            code, _ = e.orig.args
            if code in EXTEND_CODES:
                self.extend(b)
            else:
                raise e

        return self


class PostGreSQL(SQL, SQLIntoMixin):
    dbtype = 'postgresql'

    def _into(self, b, **kwargs):
        sql_stmnt = """
                    COPY {tblname} FROM '{abspath}'
                    (FORMAT {format_str}, DELIMITER E'{delimiter}',
                    NULL '{na_value}', QUOTE '{quotechar}', ESCAPE '{escapechar}',
                    HEADER {header}, ENCODING '{encoding}');"""
        sql_stmnt = sql_stmnt.format(**kwargs)
        try:
            with self.engine.begin() as conn:
                conn.execute(sql_stmnt)
        except sa.exc.NotSupportedError:
            self.extend(b)
        return self


class SQLite(SQL, SQLIntoMixin):
    dbtype = 'sqlite'

    def _into(self, b, **kwargs):
        if sys.platform == 'win32' or kwargs.get('db') == ':memory:':
            self.extend(b)
        else:
            # only to be used when table isn't already created?
            # cmd = """
            #     echo 'create table {tblname}
            #     (id integer, datatype_id integer, other_id integer);') | sqlite3 bar.db"
            #     """

            cmd = "(echo '.mode csv'; echo '.import {abspath} {tblname}';) | sqlite3 {db}"
            cmd = cmd.format(**kwargs)
            subprocess.Popen(cmd, shell=os.name != 'nt',
                             stdout=subprocess.PIPE).wait()
        return self
