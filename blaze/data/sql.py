from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from datetime import date, datetime, time
from decimal import Decimal
import sys
from dynd import nd
import sqlalchemy as sql
import sqlalchemy
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
from .csv import CSV

# http://docs.sqlalchemy.org/en/latest/core/types.html

types = {'int64': sql.types.BigInteger,
         'int32': sql.types.Integer,
         'int': sql.types.Integer,
         'int16': sql.types.SmallInteger,
         'float32': sql.types.Float(precision=24), # sqlalchemy uses mantissa
         'float64': sql.types.Float(precision=53), # for precision
         'float': sql.types.Float(precision=53),
         'real': sql.types.Float(precision=53),
         'string': sql.types.Text,
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
                 sql.types.String: 'string',
                 sql.types.Unicode: 'string',
                 sql.types.DATETIME: 'datetime',
                 sql.types.TIMESTAMP: 'datetime',
                 sql.types.FLOAT: 'float64',
                 sql.types.DATE: 'date',
                 sql.types.BIGINT: 'int64',
                 sql.types.INTEGER: 'int',
                 sql.types.Float: 'float64'})


@dispatch(sql.sql.type_api.TypeEngine)
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
        return [sql.Column(name,
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
            return sql.types.Text
        if 'U' in dshape.encoding:
            return sql.types.Unicode(length=dshape[0].fixlen)
        if 'A' in dshape.encoding:
            return sql.types.String(length=dshape[0].fixlen)
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
        self.dbtype = engine.url.drivername
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

@dispatch(SQL, CSV)
def into(sql, csv, if_exists="replace", **kwargs):
    """
    Parameters
    ----------
    if_exists : string
        {replace, append, fail}
    FORMAT : string
         Data Format
         text, csv, binary default: CSV
    DELIMITER : string
        Delimiter character
    NULL :
        Null character
        Default: '' (empty string)
    HEADER : bool
        Flag to define file contains a header (only to be used with CSV)
    QUOTE : string
        Single byte quote:  Default is double-quote. ex: "1997","Ford","E350")
    ESCAPE : string
        A single one-byte character for defining escape characters. Default is the same as the QUOTE
    ENCODING :
        An encoding name (POSTGRES): utf8, latin-1, ascii.  Default: UTF8

    ##NOT IMPLEMENTED
    # FORCE_QUOTE { ( column_name [, ...] ) | * }
    # FORCE_NOT_NULL ( column_name [, ...] )
    # OIDS : bool
    #     Specifies copying the OID for each row

    """
    dbtype = sql.engine.url.drivername
    db = sql.engine.url.database
    engine = sql.engine
    abspath = csv._abspath
    tblname = sql.tablename

    # using dialect mappings to support multiple term mapping for similar concepts
    from .dialect_mappings import dialect_terms

    def retrieve_kwarg(term):
        terms = [k for k, v in dialect_terms.iteritems() if v == term]
        for t in terms:
            val = kwargs.get(t, None)
            if val:
                return val
        return val

    for k in kwargs.keys():
        try:
            dialect_terms[k]
        except KeyError:
            raise KeyError(k, " not found in dialect mapping")

    format_str = retrieve_kwarg('format_str') or 'csv'
    encoding =  retrieve_kwarg('encoding') or ('utf8' if db=='mysql' else 'latin1')
    delimiter = retrieve_kwarg('delimiter') or csv.dialect['delimiter']
    na_value = retrieve_kwarg('na_value') or ""
    quotechar = retrieve_kwarg('quotechar') or '"'
    escapechar = retrieve_kwarg('escapechar') or quotechar
    header = retrieve_kwarg('header') or csv.header
    lineterminator = retrieve_kwarg('lineterminator') or u'\n'

    skiprows = csv.header or 0 # None or 0 returns 0
    skiprows = retrieve_kwarg('skiprows') or int(skiprows) #hack to skip 0 or 1


    copy_info = {'abspath': abspath,
                 'tblname': tblname,
                 'db': db,
                 'format_str': format_str,
                 'delimiter':delimiter,
                 'na_value': na_value,
                 'quotechar': quotechar,
                 'escapechar': escapechar,
                 'lineterminator': lineterminator,
                 'skiprows': skiprows,
                 'header': header,
                 'encoding': encoding}

    if if_exists == 'replace':
        if engine.has_table(tblname):

            # drop old table
            metadata = sqlalchemy.MetaData()
            metadata.reflect(engine, only=[tblname])
            t = metadata.tables[tblname]
            t.drop(engine)

            # create a new one
            sql.table.create(engine)


    if dbtype == 'postgresql':
        import psycopg2
        try:
            conn = sql.engine.raw_connection()
            cursor = conn.cursor()

            #lots of options here to handle formatting

            sql_stmnt = """
                        COPY {tblname} FROM '{abspath}'
                        (FORMAT {format_str}, DELIMITER E'{delimiter}',
                        NULL '{na_value}', QUOTE '{quotechar}', ESCAPE '{escapechar}',
                        HEADER {header}, ENCODING '{encoding}');
                        """
            sql_stmnt = sql_stmnt.format(**copy_info)
            cursor.execute(sql_stmnt)
            conn.commit()

        #not sure about failures yet
        except psycopg2.NotSupportedError as e:
            print("Failed to use POSTGRESQL COPY.\nERR MSG: ", e)
            print("Defaulting to sql.extend() method")
            sql.extend(csv)

    #only works on OSX/Unix
    elif dbtype == 'sqlite':
        import subprocess
        if sys.platform == 'win32':
            print("Windows native sqlite copy is not supported")
            print("Defaulting to sql.extend() method")
            sql.extend(csv)
        else:
            #only to be used when table isn't already created?
            # cmd = """
            #     echo 'create table {tblname}
            #     (id integer, datatype_id integer, other_id integer);') | sqlite3 bar.db"
            #     """

            copy_cmd = "(echo '.mode csv'; echo '.import {abspath} {tblname}';) | sqlite3 {db}"
            copy_cmd = copy_cmd.format(**copy_info)

            ps = subprocess.Popen(copy_cmd,shell=True, stdout=subprocess.PIPE)
            output = ps.stdout.read()

    elif dbtype == 'mysql':
        import MySQLdb
        try:
            conn = sql.engine.raw_connection()
            cursor = conn.cursor()

            #no null handling
            sql_stmnt = u"""
                        LOAD DATA LOCAL INFILE '{abspath}'
                        INTO TABLE {tblname}
                        CHARACTER SET {encoding}
                        FIELDS
                            TERMINATED BY '{delimiter}'
                            ENCLOSED BY '{quotechar}'
                            ESCAPED BY '{escapechar}'
                        LINES TERMINATED by '{lineterminator}'
                        IGNORE {skiprows} LINES;
                        """
            sql_stmnt = sql_stmnt.format(**copy_info)

            cursor.execute(sql_stmnt)
            conn.commit()

        #not sure about failures yet
        except MySQLdb.OperationalError as e:
            print("Failed to use MySQL LOAD.\nERR MSG: ", e)
            print("Defaulting to sql.extend() method")
            sql.extend(csv)

    else:
        print("Warning! Could not find native copy call")
        print("Defaulting to sql.extend() method")
        sql.extend(csv)
