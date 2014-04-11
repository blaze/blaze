from datetime import date, datetime, time
from decimal import Decimal
from .data_descriptor import DDesc, Capabilities
import datashape
from sqlalchemy import Column, sql
from ..utils import partition_all


# http://docs.sqlalchemy.org/en/latest/core/types.html

import sqlalchemy as alc
types = {'int64': alc.types.BigInteger,
         'int32': alc.types.Integer,
         'int': alc.types.Integer,
         'int16': alc.types.SmallInteger,
         'float': alc.types.Float,
         'string': alc.types.String,  # Probably just use only this
#         'date': alc.types.Date,
#         'time': alc.types.Time,
#         'datetime': alc.types.DateTime,
#         bool: alc.types.Boolean,
#         ??: alc.types.LargeBinary,
#         Decimal: alc.types.Numeric,
#         ??: alc.types.PickleType,
#         unicode: alc.types.Unicode,
#         unicode: alc.types.UnicodeText,
#         str: alc.types.Text,  # ??
         }

def dshape_to_alchemy(dshape):
    """

    >>> dshape_to_alchemy('int')
    <class 'sqlalchemy.sql.sqltypes.Integer'>

    >>> dshape_to_alchemy('string')
    <class 'sqlalchemy.sql.sqltypes.String'>

    >>> dshape_to_alchemy('{name: string, amount: int}')
    [Column('name', String(), table=None), Column('amount', Integer(), table=None)]
    """
    dshape = datashape.dshape(dshape)
    if str(dshape) in types:
        return types[str(dshape)]
    try:
        return [alc.Column(name, dshape_to_alchemy(typ))
                for name, typ in dshape.parameters[0].parameters[0]]
    except TypeError:
        raise NotImplementedError("Datashape not supported for SQL Schema")



class SQL_DDesc(DDesc):
    """
    A Blaze data descriptor which exposes a CSV file.

    Parameters
    ----------
    engine : A SQLAlchemy engine

    table : string
        The name of the table
    schema : datashape, datashape string, list of Columns
    """

    def __init__(self, engine, tablename, primary_key='', schema=None):
        self.engine = engine
        self.tablename = tablename

        if isinstance(schema, (str, datashape.DataShape)):
            columns = dshape_to_alchemy(schema)
            for column in columns:
                if column.name == primary_key:
                    column.primary_key = True

        if schema is None:  # Table must exist
            if not engine.has_table(tablename):
                raise ValueError('Must provide schema. Table %s does not exist'
                                 % tablename)

        self.schema = schema
        self._dshape = schema
        metadata = alc.MetaData()

        table = alc.Table(tablename, metadata, *columns)

        self.table = table
        metadata.create_all(engine)

    def __getitem__(self, key):
        pass

    def __iter__(self):
        with self.engine.connect() as conn:
            result = conn.execute(sql.select([self.table]))
            for item in result:
                yield item

    @property
    def dshape(self):
        return 'var * ' + str(self._dshape)

    @property
    def capabilities(self):
        pass

    def _extend(self, rows):
        with self.engine.connect() as conn:
            for chunk in partition_all(1000, rows):  # TODO: 1000 is hardcoded
                conn.execute(self.table.insert(), chunk)

