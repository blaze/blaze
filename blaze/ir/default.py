from blaze import prims
from operator import eq
from copy import deepcopy
from itertools import izip

#------------------------------------------------------------------------
# Phase 1 ( Types )
#------------------------------------------------------------------------

class ANode(object):

    def __init__(self, *args, **kwargs):
        n = 0
        self.__cons = {}

        for a,b in izip(self._fields, args):
            n += 1
            self.__cons[a] = b

        #for a,b in izip(self._fields[n:], kwargs):
            #self.__cons[a] = b

        if n != len(self._fields):
            raise ValueError, "Wrong number of arguments to: %s" % self.tag()
        self.init()

    def init(self):
        pass

    def to_tuple(self):
        return (self.tag(), tuple([self.__cons[k] for k in self._fields ]))

    @property
    def children(self):
        return self.__cons

    @classmethod
    def tag(cls):
        return cls.__name__

    def copy(self):
       new = deepcopy(self)
       return new

    def __eq__(self, other):
        return eqcls(self,other) and map(eq, self.__cons, other.__cons)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        kw = []
        for k in self._fields:
            kw.append("%s=%s" % (k, self.__cons[k]))
        return "%s(%s)" % (self.tag(), ",".join(kw))


class Unit(ANode):
    """
    Scalar array
    """
    pass

class Slice(ANode):
    _fields = ['start', 'stop', 'xs']


class Map(ANode):
    _fields = ['fn', 'axis', 'xs']

class Zip(ANode):
    _fields = ['fn', 'axis', 'xs', 'ys']

class Reduce(ANode):
    _fields = ['fn', 'init', 'axis', 'xs']

class Scan(ANode):
    _fields = ['fn', 'init', 'axis', 'xs']

class Permute(ANode):
    _fields = ['fn', 'xs', 'ys']

#------------------------------------------------------------------------
# Phase 2 ( Layout & Locality )
#------------------------------------------------------------------------

class ChunkedMap(object):
    pass

class ChunkedZip(object):
    pass

class ChunkedReduce(object):
    pass

class ChunkedScan(object):
    pass


class TiledMap(object):
    pass

class TiledZip(object):
    pass

class TiledReduce(object):
    pass

class TiledScan(object):
    pass

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def eqcls(a,b):
    return a.__class__ is b.__class__



if __name__ == '__main__':
    print Reduce(prims.add, 0, 3, [1,2,3])
    print Zip(prims.sqrt, 0, [1,2,3], [1,2,3])
    a = Zip(prims.sqrt, 0, [1,2,3], [1,2,3])

    assert a == a
