from collections import Set
from itertools import count
from datashape.coretypes import Var, TypeVar, Fixed

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

    An index is a mapping from a domain specification to a collection of
    byte-interfaces
    """

    def __getitem__(self, indexer):
        if isinstance(indexer, (int, str)):
            self.index1d(indexer)
        else:
            self.indexnd(indexer)

    def index1d(self, indexer):
        pass

    def indexnd(self, indexer):
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

    def __init__(self, underlying):
        self.underlying = underlying

    def size(self, ntype):
        itemsize = ntype.size()
        return Fixed(self.underlying.calculate(itemsize))

    def __repr__(self):
        return repr(self.underlying)

class Index(object):

    ordered   = False
    monotonic = False
    injective = True

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

class HierichalIndex(object):
    """
    A hierarchal index a panel with multiple index levels represented by
    tuples.

    i   j  |        A         B
    -----------------------------
    one 1  |  -0.040130 -1.471796
        2  |  -0.027718 -0.752819
        3  |  -1.166752  0.395943
    two 1  |  -1.057556 -0.012255
        2  |   1.839669  0.185417
        3  |   1.084758  0.594895

    The mapping is not ( in general ) injective. For example indexing
    into ['one'] or ['1'] will perform a SELECT like query on the tables
    contained in the level.

    Internally the representation of this would would be:

        0, 0, 1, 1, 2, 2, 3, 3
        0, 1, 0, 1, 0, 1, 0, 1

    Which endows the structure with lexicographical order.
    """

    ordered   = False
    monotonic = False
    injective = False

    def __init__(self, tuples):
        # XXX
        self.levels = 2

    def levels(self):
        return self.levels

class FilenameIndex(Index):
    """
    Take a collection of subspaces and enumerate them in the
    order of the 'name' parameter on the byte provider'. If the
    ByteProvider does not provide one, a exception is raised.

    For example a collection of files

        /003_sensor.csv
        /001_sensor.csv
        /002_sensor.csv
        /010_sensor.csv

    Would produce index:

        {
            0: File(name='/001_sensor.csv')
            1: File(name='/002_sensor.csv')
            2: File(name='/003_sensor.csv')
            3: File(name='/010_sensor.csv')
        }

    This provides a total monotonic ordering.
    """

    ordered   = True
    monotonic = True
    injective = True

    def __init__(self, space):
        try:
            self.mapping = sorted(space, self.compare)
        except KeyError:
            raise Exception("Subspace does not contain 'name' key")

    @staticmethod
    def compare(a, b):
        return a.name > b.name

class AutoIndex(Index):
    """
    Take a collection of subspaces and just create the outer
    index by enumerating them in the order passed to
    DataTable.

    Passing in a collection of Python objects would just
    enumerate the Raw ByteProviders.

        {0: Raw(ptr=31304808), 1: Raw(ptr=31305528)}

    This provides a total monotonic ordering.
    """

    ordered   = True
    monotonic = True
    injective = True

    def __init__(self, space):
        self.mapping = dict(enumerate(space.subspaces))

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
    Create a index from a Python generating function.
    """

    def __init__(self, genfn, *args):
        self.index = 0
        self.mapping = genfn(*args)

        self.monotonic = args.get('monotonic')
        self.ordered   = args.get('ordered')
        self.injective = args.get('injective')

    def __repr__(self):
        return 'arange(%r, %r)' % (
            self.start,
            self.end
        )
