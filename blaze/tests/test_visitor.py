from blaze.expr.graph import IntNode, FloatNode
from blaze.expr.visitor import BasicGraphVisitor, MroVisitor, GraphVisitor
from blaze.expr import visitor

#------------------------------------------------------------------------
# Sample Graph
#------------------------------------------------------------------------

a = IntNode(1)
b = IntNode(2)
c = FloatNode(3.0)

x = a+(b+c)
y = a+b+c

#------------------------------------------------------------------------
# Visitors
#------------------------------------------------------------------------

class Visitor(BasicGraphVisitor):

    def App(self, tree):
        return self.visit(tree.children)

    def IntNode(self, tree):
        return int

    def FloatNode(self, tree):
        return float

    def Add(self, tree):
        return self.visit(tree.children)

class MroVisitor(MroVisitor):

    def App(self, tree):
        return self.visit(tree.children)

    def Op(self, tree):
        return self.visit(tree.children)

    def Literal(self, tree):
        return True

class SkipSomeVisitor(GraphVisitor):

    def __init__(self):
        super(SkipSomeVisitor, self).__init__()
        self.found = []

    def FloatNode(self, tree):
        self.found.append(float)

class Transformer(visitor.GraphTransformer):

    def FloatNode(self, node):
        return IntNode(3)


def test_simple_visitor():
    walk = Visitor()
    a = walk.visit(x)
    assert a == [[int, [[int, float]]]]

    b = walk.visit(y)
    assert b == [[[[int, int]], float]]

def test_simple_visitor_mro():
    walk = MroVisitor()
    a = walk.visit(x)

    assert a == [[True, [[True, True]]]]

def test_graph_visitor():
    v = SkipSomeVisitor()
    v.visit(x)
    assert v.found == [float], v.found

def test_transformers():
    t = Transformer()
    x = a + (b + c)
    result = t.visit(x)

    assert Visitor().visit(result) == [[int, [[int, int]]]]

if __name__ == '__main__':
    test_transformers()
    test_simple_visitor()
    test_graph_visitor()