from abc import ABCMeta
from collections import deque
from ndtable.table import NDTable

class Node(object):
    """ Represents a node in the expression graph which Blaze compiles into
    a program for the Array VM.
    """
    # Use __slots__ so we don't incur the full cost of a class
    __slots__ = ['children', 'metadata']
    __metaclass__ = ABCMeta

    def __init__(self, *children):
        self.children  = children

    def iter_children(self):
        for child in self.children:
            yield child

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def __subclasshook__(cls, other):
        """
        If it quacks like a graph, then it is a graph.
        """
        if cls is Node:
            try:
                getattr(cls, 'name')
                getattr(cls, 'children')
                getattr(cls, '__iter__')
                return True
            except AttributeError:
                return False
        raise NotImplemented

    def __iter__(self):
        """
        Walk the graph, left to right
        """
        for name, child in self.iter_children():
            if isinstance(child, Node):
                yield child
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, Node):
                        yield item
            else:
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


# ===========
# Values
# ===========

class Literal(Node):
    __slots__ = ['children', 'metadata', 'vtype']

    def __init__(self, val):
        assert isinstance(val, self.vtype)
        self.val = val
        self.children = []

    @property
    def name(self):
        return str(self.val)

class ScalarNode(Literal):
    vtype = int

class StringNode(Literal):
    vtype = str

class IndexNode(Literal):
    vtype = tuple

    @property
    def name(self):
        return 'Index%s' % str(self.val)

# ===========
# Operations
# ===========

class Op(Node):
    __slots__ = ['children', 'metadata', 'op']

    def __init__(self, op, operands):
        self.op = op
        self.children = operands

    @property
    def name(self):
        return self.op

class NullaryOp(Op):
    arity = 0

class UnaryOp(Op):
    arity = 1

class BinaryOp(Op):
    arity = 2

class NaryOp(Op):
    arity = -1

class Slice(Op):
    # $0, start, stop, step
    arity = 4

class Apply(Op):
    pass

# ===========
# Combinators
# ===========

class Map(Op):
    pass

class Reduce(Op):
    # TODO: better word for demarcating the "fusion block"
    terminal = True

# ============
# Tree Walking
# ============

def traverse(node):
     tree = deque([node])
     while tree:
         node = tree.popleft()
         tree.extend(iter(node))
         yield node

# ===========
# Graph Nodes
# ===========

class ExprTransformer(object):

    def __init__(self):
        pass

    def visit(self, tree):
        # Visit the tree, context switching to the transform
        # function at each node.
        tree.__coiter__(self, self.transform)

    def transform(self, node):
        raise NotImplementedError()
