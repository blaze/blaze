from collections import deque, Iterable

#------------------------------------------------------------------------
# Graph Objects
#------------------------------------------------------------------------

class Node(object):
    """ Represents a node in the expression graph which Blaze compiles into
    a program for the Array VM.
    """
    # Use __slots__ so we don't incur the full cost of a class
    __slots__ = ['children']

    def __init__(self, children):
        self.children = children

    def __iter__(self):
        """
        Walk the graph, left to right
        """
        for a in self.children:
            if isinstance(a, Node):
                yield a
                for b in iter(a):
                    yield b
            else:
                import pdb; pdb.set_trace()
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

class NoTransformer(Exception):
    def __init__(self, args):
        self.args = args
    def __str__(self):
        return 'No transformer for Node: %s' % repr(self.args)

def traverse(node):
     tree = deque(node)
     while tree:
         node = tree.popleft()
         tree.extend(iter(node))
         yield node

class ExprTransformer(object):

    def __init__(self):
        pass

    # TODO: more robust!
    def visit(self, tree):
        if isinstance(tree, list):
            return [self.visit(i) for i in tree]
        else:
            nodei = tree.__class__.__name__
            trans = getattr(self,nodei, False)
            if trans:
                return trans(tree)
            else:
                return self.Unknown(tree)

    def Unknown(self, tree):
        raise NoTransformer(tree)

class MroTransformer(object):

    def __init__(self):
        pass

    def visit(self, tree):
        if isinstance(tree, list):
            return [self.visit(i) for i in tree]
        else:
            fn = None
            for o in tree.__class__.mro():
                nodei = o.__name__
                trans = getattr(self, nodei, False)
                if trans:
                    fn = trans
                    break
            if fn:
                return fn(tree)
            else:
                self.Unknown(tree)

    def Unknown(self, tree):
        raise NoTransformer(tree)
