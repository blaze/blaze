"""
Scalar layout methods for glueing together array-like structures.

If we build a structure ( A U B ) as some union of two blocks of data,
then we need only a invertible linear transformation which is able to
map coordinates between the blocks as we drill down into
subblocks::

    (i,j) -> (i', j')

Vertical Stacking
=================

::

      Block 1            Block 2

      1 2 3 4            1 2 3 4
    0 - - - -  vstack  0 * * * *
    1 - - - -          1 * * * *

              \     /

               1 2 3 4
             0 - - - - + --- Transform =
             1 - - - -       (i,j) -> (i, j)
             2 * * * * + --- Transform =
             3 * * * *       (i,j) -> (i-2, j)


Horizontal Stacking
===================

::
      Block 1            Block 2

      1 2 3 4            1 2 3 4
    0 - - - -  hstack  0 * * * *
    1 - - - -          1 * * * *

              \     /

               1 2 3 4
             0 - - * *
             1 - - * *
             2 - - * *
             3 - - * *
               |   |
               |   |
               |   + Transform =
               |     (i,j) -> (i-2, j)
               + --- Transform =
                     (i,j) -> (i, j)
"""

from copy import copy
from numpy import zeros
from functools import partial
from collections import defaultdict
from bisect import bisect_left, insort_left

# Nominal | Ordinal | Scalar

Id = lambda x:x

def ctranslate(factor, axis):
    # (i,j, ...) -> (i + a0 , j + a1, ...)

    # TODO: Certainly better way to write this...
    def T(xs):
        xs = copy(xs)
        for x, j in zip(xrange(len(xs)), axis):
            if j == 1:
                xs[x] += factor
        return xs

    def Tinv(ys):
        ys = copy(ys)
        for y, j in zip(xrange(len(ys)), axis):
            if j:
                ys[y] -= factor
        return ys

    return T, Tinv

def splitl(lst, n):
    return lst[0:n], lst[n], lst[n:-1]

def linearize(spatial):
    # create flat hash map of all indexes
    pass

def stack(c1,c2, axis):
    #   p       q        p   q
    # [1,2] | [1,2] -> [1,2,3,4]

    i1, i2 = c1.inf, c2.inf
    s1, s2 = c1.sup, c2.sup

    n = abs(i2-s1)
    assert n > 0

    T, Tinv = ctranslate(n, axis)
    return T, Tinv

class interval(object):
    """
    """
    def __init__(self, inf, sup):
        self.inf = inf
        self.sup = sup

    def __contains__(self, other):
        return self.inf <= other < self.sup

    def __iadd__(self, n):
        return interval(self.inf + n, self.sup + n)

    def __imul(self, n):
        return interval(self.inf * n, self.sup * n)

    def __iter__(self):
        yield self.inf
        yield self.sup

    def __repr__(self):
        return 'i[%i,%i]' % (self.inf, self.sup)

# Spatial
class Spatial(object):
    """
    A interval partition pointing at a Indexable reference. In C
    this would be a pointer, but not necessarily so.
    """
    def __init__(self, components, ref, tinv=None):
        self.ref = ref

        if not tinv:
            self.tinv = None
            self.transformed = False
        else:
            self.tinv = tinv
            self.transformed = True

        self.components = components

    def __contains__(self, other):
        for component in self.components:
            if other in component:
                return True
        return False

    def transform(self, t, tinv):
        coords = t(self.components)
        return Spatial( coords, self.ref, tinv)

    # Lift a set of coordinates into the "chart" space
    def inverse(self, coords):
        assert self.tinv, "Chart does not have coordinate inverse transform function"
        return self.tinv(coords)

    def __iter__(self):
        return iter(self.components)

    def __getitem__(self, indexer):
        return self.ref[indexer]

    def __repr__(self):
        return 'I(%s)' % 'x'.join(map(repr,self.components))

# Linear ( pointer-like references )
class Linear(object):
    def __init__(self, origin, ref):
        self.origin = origin
        self.ref = ref

    def __contains__(self, other):
        return

    def __getitem__(self, indexer):
        return self.mapping.get(indexer)

# Referential
class Categorical(object):
    """
    """

    def __init__(self, mapping, ref):
        self.labels  = set(mapping.keys())
        self.mapping = mapping
        self.ref = ref

    def __contains__(self, other):
        return other in self.labels

    def __getitem__(self, indexer):
        return self.mapping.get(indexer)

    def __repr__(self):
        return 'L(%r)' % list(self.labels)

class Layout(object):
    """
    Layout's build the Index objects neccessary to perform
    arbitrary getitem/getslice operations given a data layout of
    byte providers.
    """

    boundscheck = True
    # If set to False, layout will to assume that indexing operations
    # will not cause any IndexErrors to be raised

    wraparound = True
    # Allow negative indexing

    def __init__(self, partitions):
        self.points     = defaultdict(list)
        self.ppoints    = defaultdict(list)

        self.partitions = partitions
        self.top = None

        # Build up the partition search, for each dimension
        for a in partitions:
            self.top = self.top or a
            for i, b in enumerate(a.components):
                # (+1) because it's the infinum
                insort_left(self.points[i] , a)
                insort_left(self.ppoints[i], b.inf+1)

    def check_bounds(self, indexer):
        pass

    def change_coordinates(self, indexer):
        """
        Change coordinates into the memory block we're indexing
        into.
        """
        indexerl = xrange(len(indexer))

        # use xrange/len because we are mutating it
        for i in indexerl:

            idx = indexer[i]
            space = self.points[i]
            partitions = self.ppoints[i]
            size = len(space)

            # Partition index
            pdx = bisect_left(partitions, idx)

            # Edge cases
            if pdx <= 0:
                chart = space[0]
            if pdx >= size:
                chart = space[-1]
            else:
                chart = space[pdx]

            if chart.transformed:
                return chart.ref, chart.inverse(indexer)
            else:
                continue

        # Trivially partitioned
        return self.top.ref, indexer

    def __getitem__(self, indexer):
        coords = self.change_coordinates(indexer)
        return coords

    def __repr__(self):
        out = 'Layout:\n'
        for p in self.partitions:
            out += '%r -> %i\n' % (p, id(p.ref))

        out += 'Partitions:\n'
        out += repr(self.points)
        return out

# Low-level calls, never used by the end-user.
def nstack(n, a, b):
    try:
        a0, a1, a2 = splitl(a.components, n)
        b0, b1, b2 = splitl(b.components, n)
    except IndexError:
        raise Exception("Axis %i does not exist" % n)

    axis = zeros(len(a.components))
    axis[n] = 1

    T, Tinv = stack(a1, b1, list(axis))
    bT = b.transform(T, Tinv)
    assert bT.tinv

    return Layout([a, bT])

vstack = partial(nstack, 0)
hstack = partial(nstack, 1)
dstack = partial(nstack, 2)

def test_vertical_stack():
    alpha = object()
    beta  = object()

    a = interval(0,2)
    b = interval(0,2)

    x = Spatial([a,b], alpha)
    y = Spatial([a,b], beta)

    s = vstack(x,y)
    block, coords = s[[3,1]]
    assert block is beta
    assert coords == [1,1]

    block, coords = s[[0,0]]
    assert block is alpha
    assert coords == [0,0]

    block, coords = s[[1,0]]
    assert block is alpha
    assert coords == [1,0]

    block, coords = s[[2,0]]
    assert block is beta
    assert coords == [0,0]

    block, coords = s[[2,1]]
    assert block is beta
    assert coords == [0,1]

def test_horizontal_stack():
    alpha = object()
    beta  = object()

    a = interval(0,2)
    b = interval(0,2)

    x = Spatial([a,b], alpha)
    y = Spatial([a,b], beta)

    s = hstack(x,y)
    block, coords = s[[0,0]]
    assert block is alpha
    assert coords == [0,0]

    block, coords = s[[0,1]]
    assert block is alpha
    assert coords == [0,1]

    block, coords = s[[0,2]]
    assert block is beta
    assert coords == [0,0]

    block, coords = s[[2,4]]
    assert block is beta
    assert coords == [2,2]
