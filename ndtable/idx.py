from abc import ABCMeta
from numbers import Integral
from collections import Set, Mapping, Sequence
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

class CannotInfer(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return "Cannot infer"


def certain(obj, predicate):
    try:
        return getattr(obj, predicate)
    except CannotInfer:
        return False

# ================
# Indexable Spaces
# ================

class Indexable(object):
    """
    The top abstraction in the Blaze class hierarchy.

    An index is a mapping from a domain specification to a collection of
    byte or subtables.  Indexable objects can be sliced/getitemed to
    return some other object in the Blaze system.
    """

    #------------------------------------------------------------------------
    # Slice/stride/getitem interface
    #
    # Define explicit indexing operations, as distinguished from operator
    # overloads, so that we can more easily guard, test, and validate
    # indexing calls based on their semantic intent, and not merely based
    # on the type of the operand.  Such dispatch may be done in the overloaded
    # operators, but that is a matter of syntactic sugar for end-user benefit.
    #------------------------------------------------------------------------

    def slice(self, slice_obj):
        """ Extracts a subspace from this object. If there is no inner
        dimension, then this should return a scalar.  Slicing typically
        preserves the data parallelism of the slicee, and the index-space
        transform is computable in constant time.
        """
        raise NotImplementedError

    def query(self, query_expr):
        """ Queries this object and produces a view or an actual copy of
        data (or a deferred eval object which can produce those).  A query
        is typically a value-dependent streaming operation and produces
        an indeterminate number of return values.
        """
        raise NotImplementedError

    def take(self, indices, unique=None):
        """ Returns a view or copy of the indicated data.  **Indices**
        can be another Indexable or a Python iterable.  If **unique**
        if True, then implies that no indices are duplicated; if False,
        then implies that there are definitely duplicates.  If None, then
        no assumptions can be made about the indices.

        take() differs from slice() in that indices may be duplicated.
        """
        raise NotImplementedError

    #------------------------------------------------------------------------
    # Iteration protocol interface
    #
    # Defines the mechanisms by which other objects can determine the types
    # of iteration supported by this object.
    #------------------------------------------------------------------------

    def returntype(self):
        """ Returns the most efficient/general Data Descriptor this object can
        return.  Returns a value from the list the values defined in
        DataDescriptor.desctype: "buflist", "buffer", "streamlist", or
        "stream".
        """
        raise NotImplementedError
    def __index__(self):
        raise NotImplementedError()


class Space(Set):
    """
    A bag of subspaces with no notion of ordering or dimension.
    """

    def __init__(self, *subspaces):
        self.subspaces = subspaces
        self._regular = None
        self._covers  = None

    def annotate(self, regular, covers):
        self._regular = regular
        self._covers  = covers

    @property
    def regular(self):
        # Don't guess since, regular=False is not the same as "I
        # don't know yet"
        if self._regular is None:
            raise CannotInfer()
        return self._regular

    @property
    def covers(self):
        # don't guess
        if self._covers is None:
            raise CannotInfer()
        return self._covers

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
        # TODO: rethink this
        itemsize = ntype.size()

        if isinstance(itemsize, Integral):
            return Fixed(self.underlying.calculate(itemsize))
        else:
            return TypeVar('x0')

    def __repr__(self):
        return repr(self.underlying)

# =======
# Indexes
# =======

class Index(object):
    __metaclass__ = ABCMeta

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

    @classmethod
    def __subclasshook__(cls, other):
        """
        ABC Magic so that the user doesn't need to subclass Index
        in order to make a custom one.
        """
        if cls is Index:
            # Sequence of tuples kind
            if isinstance(other, Sequence):
                return True

            # Dictionary mapping to parameters of subspaces
            # (i.e. filenames, pointers, DDFS tags )
            elif isinstance(other, Mapping):
                return True

            # Arbitrary logic
            elif hasattr(other, '__index__'):
                return True

        raise NotImplemented

class HierichalIndex(Index):
    """
    A hierarchal index a panel with multiple index levels represented by
    tuples.::

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

    Internally the representation of this would would be::

        i : 0, 0, 1, 1, 2, 2, 3, 3
        j : 0, 1, 0, 1, 0, 1, 0, 1

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

    For example a collection of files::

        /003_sensor.csv
        /001_sensor.csv
        /002_sensor.csv
        /010_sensor.csv

    Would produce index::

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

        {
            0: PyObject(ptr=31304808),
            1: PyObject(ptr=31305528),
        }

    This provides a total monotonic ordering.
    """

    ordered   = True
    monotonic = True
    injective = True

    def __init__(self, shape, space):
        self.shape = shape
        self.space = space
        self.mapping = dict(enumerate(space.subspaces))

    def build(self):
        space = self.space

        if space.regular and space.covers:
            pass
        elif space.regular:
            pass

    def __index__(self):
        pass

    @property
    def start(self):
        return self.mapping[0]

    @property
    def end(self):
        return self.mapping[1]

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
