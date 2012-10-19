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
