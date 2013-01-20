from blaze.expr.graph import IntNode, FloatNode
from blaze import visitor

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

class Visitor(visitor.BasicGraphVisitor):

    def App(self, tree):
        return self.visit(tree.children)

    def IntNode(self, tree):
        return int

    def FloatNode(self, tree):
        return float

    def Add(self, tree):
        return self.visit(tree.children)

class MroVisitor(visitor.MroVisitor):

    def App(self, tree):
        return self.visit(tree.children)

    def Op(self, tree):
        return self.visit(tree.children)

    def Literal(self, tree):
        return True

class SkipSomeVisitor(visitor.GraphVisitor):

    def __init__(self):
        super(SkipSomeVisitor, self).__init__()
        self.found = []

    def FloatNode(self, tree):
        self.found.append(float)


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

#------------------------------------------------------------------------
# Transformers
#------------------------------------------------------------------------

class Transformer(visitor.GraphTransformer):

    def FloatNode(self, node):
        return IntNode(3)

class Transformer2(visitor.GraphTransformer):

    def IntNode(self, node):
        return int

    def FloatNode(self, node):
        return None

def test_transformers():
    t = Transformer()
    x = a + (b + c)
    result = t.visit(x)
    assert Visitor().visit(result) == [[int, [[int, int]]]]

    assert Transformer2().visit([a, c]) == [int]

#------------------------------------------------------------------------
# Transformers
#------------------------------------------------------------------------

class Translator(visitor.GraphTranslator):

    def App(self, tree):
        self.visitchildren(tree)
        return None

    def IntNode(self, node):
        self.result = int
        return None

    def FloatNode(self, node):
        self.result = float
        return None


def test_translators():
    t = Translator()
    assert t.visit(a + (b + c)) is None
    assert t.result == [int, [int, float]]

if __name__ == '__main__':
    test_transformers()
    test_simple_visitor()
    test_graph_visitor()
