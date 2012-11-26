from abc import ABCMeta
from numbers import Integral
from collections import Set, Mapping, Sequence
from itertools import count
from datashape.coretypes import Var, TypeVar, Fixed

"""
An index describes how we map the entire "space" of indexer objects to
the range of storage blocks that comprise the NDArray.
"""

class CannotCast(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return "Cannot infer"

def certain(obj, predicate):
    try:
        return getattr(obj, predicate)
    except CannotCast:
        return False

#------------------------------------------------------------------------
# Indexable
#------------------------------------------------------------------------

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
        """ Extracts a subset of values from this object. If there is
        no inner dimension, then this should return a scalar. Slicing
        typically preserves the data parallelism of the slicee, and the
        index-space transform is computable in constant time.
        """
        raise NotImplementedError

    def query(self, query_expr):
        """ Queries this object and produces a view or an actual copy
        of data (or a deferred eval object which can produce those). A
        query is typically a value-dependent streaming operation and
        produces an indeterminate number of return values.
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

#------------------------------------------------------------------------
# Space
#------------------------------------------------------------------------

# The unordered conglomeration of spaces that comprise a indexable
# object. This is endowed with structure from the parent object
# (NDArray, NDTable). it can be iterated over but has no intrinsic
# order. It merely holds references to the spaces that the charts of
# indexes use can dig into when given high-level coordinate objects.

class Space(Set):
    """
    A bag of subspaces with no notion of ordering or dimension.
    """

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


class AutoIndex(Index):
    """
    Take a collection of subspaces and just create the outer
    index by enumerating them.
    """

    ordered   = True
    monotonic = True
    injective = True

    def __init__(self, shape, space):
        self.shape = shape
        self.space = space
        self.mapping = dict(enumerate(space.subspaces))

    @property
    def start(self):
        return self.mapping[0]

    @property
    def end(self):
        return self.mapping[1]
