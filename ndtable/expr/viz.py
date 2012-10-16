import pydot
from cStringIO import StringIO
from ndtable.expr.nodes import Op
from collections import Counter
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile

def dump(node, ipython=True, filename=None):
    """
    Dump the expression graph to either a file or IPython.
    """
    _, graph = build_graph(node)
    if filename:
        return view(filename, graph)
    elif ipython:
        return browser(graph)

def build_graph(node, graph=None, context=None, tree=False):
    top = pydot.Node( node.name )

    if not graph:
        graph = pydot.Graph(graph_type='digraph')
        graph.add_node( top )
        context = Counter()

    for child in node.children:
        nd, _ = build_graph(child, graph, context)

        # Ensure the graph is a tree by adding numbers to the
        # labels of nodes.
        if tree:

            if context[nd.name] == 0:
                # a
                name = nd.name
            else:
                # a1, a2, a3
                name = nd.name + str(context[nd.name])

            context[nd.name] += 1

        # Allow cycles or "level-jumping" graphs
        else:
            name = nd.name

        nd = pydot.Node(name)
        graph.add_node( nd )
        graph.add_edge( pydot.Edge(top, nd) )

    return node, graph

def browser(graph):
    from IPython.core.display import Image

    dotstr = graph.to_string()

    with NamedTemporaryFile(delete=True) as tempdot:
        tempdot.write(dotstr)
        tempdot.flush()
        p = Popen(['dot','-Tpng',tempdot.name] ,stdout=PIPE)

        pngdata = StringIO(p.communicate()[0]).read()

    return Image(data=pngdata)

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
