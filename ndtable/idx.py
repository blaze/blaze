"""
An index describes how we map the entire "space" of indexer objects to
the range of storage blocks that comprise the NDArray.

    special  = Any | All
    atom     = int | str | special
    indexer  = atom | slice [atom]

The index may span multiple regions of data. Hypothetically we
may have a table comprised of::

        a    b    c    d
      +------------------+
    x |                  |
      |     HDF5         |
    y |                  |
      +------------------+
    z |     CSV          |
      +------------------+

A slice across a domain could possibly span multiple regions::

        a    b    c    d
      +------------------+
    x |                  |
      |           *      |
    y |           *      |
      +-----------*------+
    z |           *      |
      +------------------+

Or even more complicated in higher-dimensional cases.

"""

from functools import total_ordering
from numpy import arange, searchsorted

class Region(object):
    def __init__(self, adaptor):
        pass

class Index(object):

    ordered   = False
    monotonic = False

    def __init__(self, byte_interfaces):
        self.byte_interfaces = byte_interfaces

    def as_contigious(self):
        pass

    def as_strided(self):
        pass

    def as_stream(self):
        pass

    def __or__(self, other):
        return self.union(self, other)

    def union(self, other):
        return

class AutoIndex(Index):

    def __init__(self, byte_interfaces):
        # arange
        pass
