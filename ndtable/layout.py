"""
Scalar layout methods for glueing together array-like structures.

If we build a structure ( A U B ) as some union of two blocks of data,
then we need only a invertible linear transformation which is able to
map coordinates between the blocks as we drill down into subblocks.

    (i,j) -> (i', j')

"""

from copy import copy
from numpy import zeros
from functools import partial
from collections import defaultdict
from bisect import bisect_left, insort_left

# Nominal | Ordinal | Scalar

def ctranslate(factor, axis):
    # (i,j, ...) -> (i + a0 , j + a1, ...)

    # TODO: Certainly better way to write this...
    def T(xs):
        xs = copy(xs)
        for x, j in zip(xrange(len(xs)), axis):
            if j == 1:
                print xs
                xs[x] += factor
        return xs

    def Tinv(ys):
        ys = copy(ys)
        for y, j in zip(xrange(len(ys)), axis):
            if j:
                ys[y] -= factor
        return ys

    return T, Tinv

def index(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

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
    A interval partition pointing at a indexable reference.
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

    def untransform(self):
        coords = self.tinv(self.components)
        return Spatial( coords )

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
    A set of key references pointing at a indexable reference.
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

    def __init__(self, partitions):
        self.points = defaultdict(list)
        self.partitions = partitions

        # Build up the partition search, for each dimension
        for a in partitions:
            for i, b in enumerate(a.components):
                insort_left(self.points[i], b.inf)


    def __getitem__(self, indexer):
        access = []
        for i, idx in enumerate(indexer):
            space = self.points[i]
            # Partition number
            pdx = bisect_left(space, idx)

            if pdx == len(self.points[i]):
                # This dimension is not partitioned
                access += [idx]
            else:
                # This dimension is partitioned
                access += [pdx]

        return access

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

    return Layout([a, bT])

vstack = partial(nstack, 0)
hstack = partial(nstack, 1)
dstack = partial(nstack, 2)

def test_simple():
    alpha = object()
    beta  = object()

    a = interval(0,2)
    b = interval(0,2)

    x = Spatial([a,b], alpha)
    y = Spatial([a,b], beta)

    stacked = vstack(x,y)

    result = stacked[(3,1)]
    import pdb; pdb.set_trace()
