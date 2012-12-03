from ndtable.expr.graph import IntNode, FloatNode
from ndtable.expr.visitor import ExprVisitor, MroVisitor

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

class Visitor(ExprVisitor):

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

def test_simple_transform():
    walk = Visitor()
    a = walk.visit(x)
    assert a == [[int, [[int, float]]]]

    b = walk.visit(y)
    assert b == [[[[int, int]], float]]

def test_simple_transform_mro():
    walk = MroVisitor()
    a = walk.visit(x)

    assert a == [[True, [[True, True]]]]
