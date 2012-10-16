from collections import deque
from ndtable.table import NDTable

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

    @property
    def name(self):
        return 'GenericNode'

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
        child_walk = self.iter_children()
        children = dict(enumerate(child_walk)).viewitems()

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

#------------------------------------------------------------------------
# Traversal
#------------------------------------------------------------------------

def traverse(node):
     tree = deque(node)
     while tree:
         node = tree.popleft()
         tree.extend(iter(node))
         yield node

class ExprTransformer(object):

    def __init__(self):
        pass

    def visit(self, tree):
        # Visit the tree, context switching to the transform
        # function at each node.
        tree.__coiter__(self, self.transform)

    def transform(self, node):
        raise NotImplementedError()
