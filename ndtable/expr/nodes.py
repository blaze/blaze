from collections import deque
from ndtable.table import NDTable

class Node(object):
    """ Represents a node in the expression graph which Blaze compiles into
    a program for the array VM.
    """
    # Use __slots__ so we don't incur the full cost of a class
    __slots__ = ['fields', 'metadata', 'listeners']

    def __init__(self, *fields):
        self.fields = fields
        self.listeners = []

    def iter_fields(self):
        for field in self.fields:
            yield field

    @property
    def name(self):
        if self.fields:
            return self.__class__.__name__  +  " " + str(self.fields[0])
        else:
            return self.__class__.__name__

    def depends_on(self, *depends):
        self.listeners += depends

    attach = depends_on

    def __iter__(self):
        for name, field in self.iter_fields():
            if isinstance(field, Node):
                yield field
            elif isinstance(field, list):
                for item in field:
                    if isinstance(item, Node):
                        yield item

    def __coiter__(self, co):
        """
        Tree transformer using coroutines and views.
        """
        fields = dict(enumerate(self.fields)).viewitems()

        def switch(field):
            changed = co.send(field)
            if changed:
                fields[idx] = changed
            else:
                del fields[idx]

        for idx, field in fields:

            if isinstance(field, Node):
                switch(field)

            elif isinstance(field, list):
                for item in field:
                    if isinstance(item, Node):
                        switch(field)




# ===========
# Values
# ===========

class Literal(Node):
    pass

class ScalarNode(Literal):
    pass

class StringNode(Literal):
    pass

# ===========
# Operations
# ===========

class Op(Node):
    pass

class UnaryOp(Op):
    pass

class BinaryOp(Op):
    pass

class NaryOp(Op):
    pass

class Slice(Op):
    pass

class Apply(Op):
    pass

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

class ExprGraph(object):

    def __init__(self, top):
        self.top = top

        # NDTable | Table, so that we maintain closure
        # throughout the expression evaluation
        self.target = NDTable

    def eval(self):
        # Submit to execution engine!
        pass

    def __iter__(self):
        return traverse(self.top)

    def to_dot(self):
        # graphviz stuff
        pass

class ExprTransformer(object):

    def __init__(self):
        pass

    def visit(self, tree):
        # Visit the tree, context switching to the transform
        # function at each node.
        tree.__coiter__(self, self.transform)

    def transform(self, node):
        raise NotImplementedError()
