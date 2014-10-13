from __future__ import absolute_import, division, print_function

from .regex import RegexDispatcher
from .dispatch import dispatch

__all__ = 'resource', 'create_index', 'drop'

resource = RegexDispatcher('resource')

@resource.register('.*', priority=1)
def resource_all(uri, *args, **kwargs):
    raise NotImplementedError("Unable to parse uri to data resource: " + uri)


@resource.register('.+::.+', priority=15)
def resource_split(uri, *args, **kwargs):
    uri, other = uri.rsplit('::', 1)
    return resource(uri, other, *args, **kwargs)


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
