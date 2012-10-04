from collections import Set
from itertools import count
from datashape.coretypes import Var, TypeVar

"""
An index describes how we map the entire "space" of indexer objects to
the range of storage blocks that comprise the NDArray.
from datashape.coretypes import TypeVar

    special  = Any | All
    atom     = int | str | special
    indexer  = atom | slice [atom]

The index may span multiple subspaces of data. Hypothetically we may
have a table comprised of::

        a    b    c    d
      +------------------+
    x |                  |
      |     HDF5         |
    y |                  |
      +------------------+
    z |     CSV          |
      +------------------+

A slice across a domain could possibly span multiple subspaces::

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

class Indexable(object):
    """
    The top abstraction in the Blaze class hierarchy.
    """

    def __getitem__(self, indexer):
        pass

    def __getslice__(self, start, stop, step):
        pass


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

    ordered   = True
    monotonic = True

    def __init__(self, size):
        #self.mapping = arange(0, size)
        self.mapping = range(size)

    @property
    def start(self):
        self.mapping[0]

    @property
    def end(self):
        self.mapping[1]

    def __repr__(self):
        return 'arange(%r, %r)' % (
            self.start,
            self.end
        )

class GeneratingIndex(Index):
    """
    Create a index from a Ptyhon generating function.
    """

    def __init__(self, genfn, *args):
        self.index = 0
        self.mapping = genfn(*args)

    def __next__(self):
        return next(self.mapping)

    def __repr__(self):
        return 'arange(%r, %r)' % (
            self.start,
            self.end
        )

def can_coerce(axis, idx_size, adapt):
    # Can we meaningfully interprete the axis as providing a
    # bijection between elements of the adaptor byte descriptor
    # in the context of the given element size.

    if isinstance(axis, Var):
        assert axis.lower < idx_size < axis.upper
        return AutoIndex(idx_size)
    if isinstance(axis, TypeVar):
        # Spin off a unique generator to map to elements on this
        # dimension.
        return GeneratingIndex(count, 0)
    else:
        raise NotImplementedError()
