"""
Scalar layout methods for glueing together array-like structures.
"""

from idx import AutoIndex

# Nominal | Ordinal | Scalar

def splitl(lst, n):
    return lst[0:n], lst[n], lst[n:-1]

def linearize(spatial):
    # create flat hash map of all indexes
    pass

def stack(c1,c2):
    #   p       q        p   q
    # [1,2] | [1,2] -> [1,2,3,4]

    i1, i2 = c1.inf, c2.inf
    s1, s2 = c1.sup, c2.sup

    n = abs(i2-s1)
    assert n > 0
    return c1, c2.scale(n)

class interval(object):
    def __init__(self, inf, sup):
        self.inf = inf
        self.sup = sup

    def __contains__(self, other):
        return self.inf <= other < self.sup

    def scale(self, n):
        return interval(self.inf + n, self.sup + n)

    def __repr__(self):
        return '[%i,%i]' % (self.inf, self.sup)

# Spatial
class Spatial(object):
    """
    A interval partition pointing at a indexable reference.
    """
    def __init__(self, components, ref):
        self.components = components
        self.ref = ref

    def __contains__(self, other):
        for component in self.components:
            if other in component:
                return True
        return False

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
        self.partitions = partitions

    def __getitem__(self, indexer):
        components = list(enumerate(indexer))

    def __repr__(self):
        out = 'Layout:\n'
        for p in self.partitions:
            out += '%r -> %i\n' % (p, id(p.ref))
        return out

# Low-level calls, never used by the end-user.
def hstack(a,b):
    a0, a1, a2 = splitl(a.components, 0)
    b0, b1, b2 = splitl(b.components, 0)

    # stack the first axi
    x1, y1 = stack(a1,b1)

    p1 = Spatial(a0 + [x1] + a2, a.ref)
    p2 = Spatial(b0 + [y1] + b2, b.ref)

    return Layout([p1, p2])

def vstack(a,b):
    a0, a1, a2 = splitl(a.components, 1)
    b0, b1, b2 = splitl(b.components, 1)

    # stack the first axi
    x1, y1 = stack(a1,b1)

    p1 = Spatial(a0 + [x1] + a2, a.ref)
    p2 = Spatial(b0 + [y1] + b2, b.ref)

    return Layout([p1, p2])

def test_simple():
    alpha = object()
    beta = object()

    a = interval(0,1)
    b = interval(0,1)

    x = Spatial([a,b], alpha)
    y = Spatial([a,b], beta)

    #stacked = hstack(x,y)
    stacked = vstack(x,y)
    print stacked
    print stacked[(1,1)]
