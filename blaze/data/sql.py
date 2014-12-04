from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import warnings
from toolz import first
from dynd import nd
import sqlalchemy as sql
import sqlalchemy
import datashape
from datashape import dshape, var, Record, Option, isdimension, DataShape
from datashape.predicates import isrecord
from itertools import chain
import subprocess
from multipledispatch import MDNotImplementedError
import gzip
from collections import Iterator

from ..dispatch import dispatch
from ..utils import partition_all
from .core import DataDescriptor
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
                 sql.types.NUMERIC: 'float64',  # TODO: extend datashape to decimal
                 sql.types.BIGINT: 'int64',
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


@dispatch(sql.engine.base.Engine)
def discover(engine):
    metadata = sql.MetaData()
    metadata.reflect(engine)
    pairs = []
    for name, table in sorted(metadata.tables.items(), key=first):
        try:
            pairs.append([name, discover(table)])
        except sqlalchemy.exc.CompileError as e:
            print("Can not discover type of table %s.\n" % name +
                "SQLAlchemy provided this error message:\n\t%s" % e.message +
                "\nSkipping.")
        except NotImplementedError as e:
            print("Blaze does not understand a SQLAlchemy type.\n"
                "Blaze provided the following error:\n\t%s" % e.message +
                "\nSkipping.")
    return DataShape(Record(pairs))


def dshape_to_table(name, ds, metadata=None):
    """
    Create a SQLAlchemy table from a datashape and a name

    >>> dshape_to_table('bank', '{name: string, amount: int}') # doctest: +NORMALIZE_WHITESPACE
    Table('bank', MetaData(bind=None),
          Column('name', Text(), table=<bank>, nullable=False),
          Column('amount', Integer(), table=<bank>, nullable=False),
          schema=None)
    """

    if isinstance(ds, _strtypes):
        ds = dshape(ds)
    metadata = metadata or sql.MetaData()
    cols = dshape_to_alchemy(ds)
    return sql.Table(name, metadata, *cols)


@dispatch(object, _strtypes)
def create_from_datashape(o, ds, **kwargs):
    return create_from_datashape(o, dshape(ds), **kwargs)

@dispatch(sql.engine.base.Engine, DataShape)
def create_from_datashape(engine, ds, **kwargs):
    assert isrecord(ds)
    metadata = sql.MetaData(engine)
    for name, sub_ds in ds[0].dict.items():
        t = dshape_to_table(name, sub_ds, metadata=metadata)
        t.create()
    return engine


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
    if isinstance(dshape, datashape.DateTime):
        if dshape.tz:
            return sql.types.DateTime(timezone=True)
        else:
            return sql.types.DateTime(timezone=False)
    raise NotImplementedError("No SQLAlchemy dtype match for datashape: %s"
            % dshape)


@dispatch(Iterator, sql.Table)
def into(_, t, **kwargs):
    engine = t.bind
    with engine.connect() as conn:
        result = conn.execute(sql.sql.select([t]))
        for item in result:
            yield item


@dispatch((list, tuple, set), sql.Table)
def into(a, t, **kwargs):
    if not isinstance(a, type):
        a = type(a)
    return a(into(Iterator, t, **kwargs))


@dispatch(sql.Table, Iterator)
def into(t, rows, **kwargs):
    assert not isinstance(t, type)
    rows = iter(rows)

    # We see if the sequence is of tuples or dicts
    # If tuples then we coerce them to dicts
    try:
        row = next(rows)
    except StopIteration:
        return
    rows = chain([row], rows)
    if isinstance(row, (tuple, list)):
        names = discover(t).measure.names
        rows = (dict(zip(names, row)) for row in rows)

    engine = t.bind
    with engine.connect() as conn:
        for chunk in partition_all(1000, rows):  # TODO: 1000 is hardcoded
            conn.execute(t.insert(), chunk)

    return t


@dispatch(sql.Table, (list, tuple, set, DataDescriptor))
def into(t, rows, **kwargs):
    return into(t, iter(rows), **kwargs)


@dispatch(sql.Table, CSV)
def into(sql, csv, if_exists="append", **kwargs):
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
    if csv.open == gzip.open:
        raise MDNotImplementedError()
    engine = sql.bind
    dbtype = engine.url.drivername
    db = engine.url.database
    tblname = sql.name

    abspath = csv._abspath

    # using dialect mappings to support multiple term mapping for similar concepts
    from .dialect_mappings import dialect_terms

    def retrieve_kwarg(term):
        terms = [k for k, v in dialect_terms.items() if v == term]
        for t in terms:
            val = kwargs.get(t, None)
            if val:
                return val
        return val

    format_str = retrieve_kwarg('format_str') or 'csv'
    encoding =  retrieve_kwarg('encoding') or ('utf8' if db=='mysql' else 'latin1')
    delimiter = retrieve_kwarg('delimiter') or csv.dialect['delimiter']
    na_value = retrieve_kwarg('na_value') or ""
    quotechar = retrieve_kwarg('quotechar') or '"'
    escapechar = retrieve_kwarg('escapechar') or quotechar
    header = retrieve_kwarg('header') or csv.header
    lineterminator = retrieve_kwarg('lineterminator') or u'\r\n'

    skiprows = csv.header or 0 # None or 0 returns 0
    skiprows = retrieve_kwarg('skiprows') or int(skiprows) #hack to skip 0 or 1

    mysql_local = retrieve_kwarg('local') or ''

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
                 'encoding': encoding,
                 'mysql_local': mysql_local}

    if if_exists == 'replace':
        if engine.has_table(tblname):

            # drop old table
            sql.drop(engine)

            # create a new one
            sql.create(engine)


    if dbtype == 'postgresql':
        import psycopg2
        try:
            conn = engine.raw_connection()
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
            print("Defaulting to stream through Python.")
            raise MDNotImplementedError()

    #only works on OSX/Unix
    elif dbtype == 'sqlite':
        if db == ':memory:':
            raise MDNotImplementedError()
        elif sys.platform == 'win32':
            warnings.warn("Windows native sqlite copy is not supported\n"
                          "Defaulting to stream through Python.")
            raise MDNotImplementedError()
        else:
            #only to be used when table isn't already created?
            # cmd = """
            #     echo 'create table {tblname}
            #     (id integer, datatype_id integer, other_id integer);') | sqlite3 bar.db"
            #     """

            copy_cmd = "(echo '.mode csv'; echo '.import {abspath} {tblname}';) | sqlite3 {db}"
            copy_cmd = copy_cmd.format(**copy_info)

            ps = subprocess.Popen(copy_cmd, shell=True, stdout=subprocess.PIPE)
            output = ps.stdout.read()

    elif dbtype == 'mysql':
        import MySQLdb
        try:
            conn = engine.raw_connection()
            cursor = conn.cursor()

            #no null handling
            sql_stmnt = u"""
                        LOAD DATA {mysql_local} INFILE '{abspath}'
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
            print("Defaulting to stream through Python.")
            raise MDNotImplementedError()

    else:
        print("Warning! Could not find native copy call")
        print("Defaulting to stream through Python.")
        raise MDNotImplementedError()

    return sql
