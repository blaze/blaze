from ndtable.engine.pipeline import toposort, Pipeline
from ndtable.expr.graph import IntNode
from ndtable.expr.nodes import ExprTransformer, MroTransformer
from ndtable.expr.graph import FloatNode


#------------------------------------------------------------------------
# Sample Graph
#------------------------------------------------------------------------

a = IntNode(1)
b = IntNode(2)
c = FloatNode(3.0)

x = a+(b+c)
y = a+b+c

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

class Visitor(ExprTransformer):

    def App(self, tree):
        return self.visit(tree.children)

    def add(self, tree):
        return self.visit(tree.children)

    def IntNode(self, tree):
        return int

    def FloatNode(self, tree):
        return float

class MroVisitor(MroTransformer):

    def object(self, tree):
        return self.visit(tree.children)

    def Literal(self, tree):
        return True

def test_simple_sort():
    lst = toposort(x)
    assert len(lst) == 6

def test_simple_pipeline():
    a = toposort(x)

    line = Pipeline()
    output = line.run_pipeline(a)

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
