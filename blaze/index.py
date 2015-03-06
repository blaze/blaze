from .dispatch import dispatch
from .compatibility import basestring
from odo import resource

@dispatch(object, (basestring, list, tuple))
def create_index(t, column_name_or_names, name=None):
    """Create an index on a column.

    Parameters
    ----------
    o : table-like
    index_name : str
        The name of the index to create
    column_name_or_names : string, list, tuple
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


@dispatch(basestring, (basestring, list, tuple))
def create_index(uri, column_name_or_names, name=None, **kwargs):
    data = resource(uri, **kwargs)
    create_index(data, column_name_or_names, name=name)
    return data

