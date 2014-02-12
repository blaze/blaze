"""
SQL data descriptor using pyodbc.
"""

from __future__ import absolute_import, division, print_function

from ...datadescriptor import IDataDescriptor, Capabilities
from .query import execute

class SQLDataDescriptor(IDataDescriptor):
    """
    SQL data descriptor.
    """

    deferred = True

    def __init__(self, dshape, table, conn):
        """
        Parameters
        ----------

        table: str
            Holds an SQL table name from which we can select data. This may also
            be some other valid query on which we can do further selection etc.
        """
        self._dshape = dshape
        self.table = table
        self.conn = conn

        # TODO: Validate query as a suitable expression to select from

    @property
    def strategy(self):
        return 'sql'

    @property
    def dshape(self):
        return self._dshape

    @property
    def capabilities(self):
        """The capabilities for the SQL data descriptor."""
        return Capabilities(
            immutable = True,
            deferred = False,
            persistent = True,
            appendable = False,
            remote=True,
            )

    def __iter__(self):
        return iter(execute(self.conn, self.dshape,
                            "select * from ?", (self.table,)))

    def __getitem__(self, item):
        raise NotImplementedError

    def dynd_arr(self):
        raise NotImplementedError

    def __repr__(self):
        return "SQLDataDescriptor(%s)" % (self.table,)

    def __str__(self):
        arrname = str(self.query)
        sdb_array = self.conn.wrap_array(arrname)
        return str(sdb_array.toarray())

    _printer = __str__


