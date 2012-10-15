from collections import deque

class Node(object):
    __slots__ = ['fields', 'metadata']

    def __init__(self, *fields):
        self.fields = fields
        self.listeners = []

    def iter_fields(self):
        for field in self.fields:
            yield field

    def depends_on(self, depends):
        self.listeners += depends

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

        # DataTable | Table, so that we maintain closure
        # throughout the expression evaluation
        self.target = DataTable

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

# ===========
# Values
# ===========

class Val(Node):
    pass

class ScalarNode(Val):
    pass

class StringNode(Val):
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
