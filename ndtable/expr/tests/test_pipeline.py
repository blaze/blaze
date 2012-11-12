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
# Tests
#------------------------------------------------------------------------

OP  = 0
APP = 1
VAL = 2

def test_simple_sort():
    lst = toposort(lambda x: True, x)
    assert len(lst) == 6

def test_simple_sort_ops():
    from itertools import ifilter
    res = ifilter(lambda x: x.kind, x)
    import pdb; pdb.set_trace()

    lst = toposort(lambda x: x.kind == 0, x)
    import pdb; pdb.set_trace()
    assert len(lst) == 3

def test_simple_pipeline():
    line = Pipeline()
    plan = line.run_pipeline(x)
    print plan
