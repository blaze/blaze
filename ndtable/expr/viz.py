import os
import pydot
from ndtable.expr.nodes import Op, ScalarNode, StringNode
from subprocess import Popen
from tempfile import NamedTemporaryFile

def build_graph(node, graph=None):
    top = pydot.Node( node.name )

    if not graph:
        graph = pydot.Graph(graph_type='digraph')
        graph.add_node( top )

    for child in node.children:
        nd, _ = build_graph(child, graph)
        nd = pydot.Node(nd.name)
        graph.add_node( nd )
        graph.add_edge( pydot.Edge(top, nd) )

    return node, graph

def view(fname, graph):
    dotstr = graph.to_string()

    with NamedTemporaryFile(delete=True) as tempdot:
        tempdot.write(dotstr)
        tempdot.flush()

        p = Popen(['dot','-Tpng',tempdot.name,'-o','%s.png' % fname])
        p.wait()
        assert p.returncode == 0

        # Linux
        p = Popen(['feh', '%s.png' % fname])
        # Macintosh
        #p = Popen(['open', fname])

def test_simple():

    a = Op('add')
    b = ScalarNode(1)
    c = ScalarNode(2)
    a.attach(b,c)

    _, graph = build_graph(a)
    view('s1', graph)

def test_nested():

    a = Op('add')
    b = ScalarNode(1)
    c = ScalarNode(2)

    a.attach(b,c)

    d = StringNode('spock')
    e = StringNode('kirk')

    c.attach(d,e)

    _, graph = build_graph(a)
    view('nested', graph)

def test_complex():

    a = Op('add')

    w = ScalarNode(0)
    x = ScalarNode(1)
    y = ScalarNode(2)
    z = ScalarNode(3)

    b = Op('xor')
    c = Op('and')

    a.attach(b,c)
    b.attach(w,x)
    c.attach(y,z)

    _, graph = build_graph(a, None)
    view('complex', graph)
