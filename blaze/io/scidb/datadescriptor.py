"""
SciDB data descriptor.
"""

from __future__ import absolute_import, division, print_function

from blaze.datadescriptor import DDesc


class SciDB_DDesc(DDesc):
    """
    SciDB data descriptor.
    """

    deferred = True

    def __init__(self, dshape, query, conn):
        """
        Parameters
        ----------

        query: Query
            Query object signalling the SciDB array to be referenced, or the
            (atomic) expression to construct an array.
        """
        self.query = query
        self._dshape = dshape
        self.conn = conn

    @property
    def strategy(self):
        return 'scidb'

    @property
    def dshape(self):
        return self._dshape

    @property
    def capabilities(self):
        """The capabilities for the scidb data descriptor."""
        return {'immutable': True,
                'deferred': False,
                'persistent': True,
                'appendable': False}

    # TODO: below

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def dynd_arr(self):
        raise NotImplementedError

    def __repr__(self):
        return "SciDBDesc(%s)" % (str(self.query),)

    def __str__(self):
        arrname = str(self.query)
        sdb_array = self.conn.wrap_array(arrname)
        return str(sdb_array.toarray())

    _printer = __str__
