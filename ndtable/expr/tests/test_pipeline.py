from ndtable.engine.pipeline import toposort, Pipeline
from ndtable.expr.graph import IntNode, FloatNode
from ndtable.expr.visitor import ExprTransformer, MroTransformer, ExprPrinter,\
    MorphismPrinter

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

class Visitor(ExprTransformer):

    def App(self, tree):
        return self.visit(tree.children)

    def IntNode(self, tree):
        return int

    def FloatNode(self, tree):
        return float

    def add(self, tree):
        return self.visit(tree.children)

class MroVisitor(MroTransformer):

    def App(self, tree):
        return self.visit(tree.children)

    def Op(self, tree):
        return self.visit(tree.children)

    def Literal(self, tree):
        return True

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

def test_simple_sort():
    lst = toposort(x)
    assert len(lst) == 6

# Experimental
def test_simple_pipeline():
    a = toposort(x)

    line = Pipeline()

    #output = line.run_pipeline(a)

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


# commented because they have IO effects that pollute the test
# output

# def test_printer():
#     walk = ExprPrinter()
#     walk.visit(x)
#     walk.visit(y)
#
# def test_mprinter():
#     walk = MorphismPrinter()
#     walk.visit(x)
#     walk.visit(y)
