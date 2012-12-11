from copy import copy
from collections import deque

from blaze.engine import pipeline

#------------------------------------------------------------------------
# Graph Objects
#------------------------------------------------------------------------

class Node(object):
    """ Represents a node in the expression graph which Blaze compiles into
    a program for execution engine.
    """
    # Use __slots__ so we don't incur the full cost of a class
    __slots__ = ['children']
    _fields = ['children']

    def __init__(self, children):
        self.children = children

    def eval(self):
        """ Evaluates the expression graph """
        from blaze.rts.execution import execplan

        # setup a default pipeline
        line = pipeline.Pipeline()

        # generate the plan
        ctx, plan = line.run_pipeline(self)

        # submit to the runtime for the result
        return execplan(ctx, plan)

    def __iter__(self):
        """ Walk the graph, left to right """
        for a in self.children:
            if isinstance(a, Node):
                yield a
                for b in iter(a):
                    yield b
            elif a is None:
                pass
            else:
                raise TypeError('Invalid children')

    def __coiter__(self, co):
        """
        Walk the graph, left to right with a coroutine.
        """
        children = dict(enumerate(self)).viewitems()

        def switch(child):
            changed = co.send(child)
            if changed:
                children[idx] = changed
            else:
                del children[idx]

        for idx, child in children:

            if isinstance(child, Node):
                switch(child)

            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, Node):
                        switch(child)

    #def __hash__(self):
        ## tree hashing, xor with the hash of children
        #h = hash(type(self))
        #for child in self.children:
            #h ^= hash(child)
        #return h

    @property
    def name(self):
        return 'GenericNode'

#------------------------------------------------------------------------
# Traversal
#------------------------------------------------------------------------

def traverse(node):
     tree = deque(node)
     while tree:
         node = tree.popleft()
         tree.extend(iter(node))
         yield node

#------------------------------------------------------------------------
# Coiterators
#------------------------------------------------------------------------

def coiter(obj, co):
    """ Prefix form of __coiter__ """
    return obj.__coiter__(co)

def flat(tree, depth=0):
    """
    Flatten a non-cyclic iterable

    Usage:
        flatten([[[[1,2],3],[4,5]]) = [1,2,3,4,5]
    """
    try:
        for x in tree:
            for y in flat(x, depth+1):
                yield y
    except TypeError:
        yield tree

#------------------------------------------------------------------------
# Functors
#------------------------------------------------------------------------

#          fmap f
#    a              f(a)
#   / \              / \
#  b   c     =    f(b) f(c)
#     /  \             /  \
#    d    e         f(d)   f(e)

def fmap(f, tree):
    """ Functor for trees """
    # this is trivial because all nodes use __slots__
    x1 = copy(tree)
    for x in tree:
        x1.children = fmap(f, x1.children)
    return x1
