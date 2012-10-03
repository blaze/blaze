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
from collections import Set

# TODO: What to name the container of subspaces?
class Space(Set):

    def __init__(self, *subspaces):
        self.subspaces = subspaces

    def __contains__(self, other):
        return other in self.subspaces

    def __iter__(self):
        return iter(self.subspaces)

    def __len__(self):
        return len(self.subspaces)

    def __repr__(self):
        out = ''
        for space in self.subspaces:
            out += 'Subspace: %r\n' % space
        return out

class Subspace(object):

    def __init__(self, adaptor):
        self.adaptor = adaptor

    def size(self, ntype):
        itemsize = ntype.size()
        return self.adaptor.calculate(itemsize)

    def __repr__(self):
        return 'Region(%r)' % self.adaptor

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

    def __init__(self, size):
        #self.mapping = arange(0, size)
        self.mapping = range(size)

    def start(self):
        pass

    def end(self):
        pass

    def __repr__(self):
        return 'arange(%r, %r)' % (
            self.mapping[0],
            self.mapping[-1]
        )
