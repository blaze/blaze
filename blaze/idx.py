from abc import ABCMeta
from numbers import Integral
from collections import Set, Mapping, Sequence
from itertools import count
from datashape import Fixed, dynamic

from numpy import searchsorted, lexsort

"""
An index describes how we map the entire "space" of indexer objects to
the range of storage blocks that comprise the NDArray.
"""

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
            return dynamic

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
