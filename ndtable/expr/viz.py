import pydot

from cStringIO import StringIO
from collections import Counter
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile

def dump(node, ipython=True, filename=None, tree=False):
    """
    Dump the expression graph to either a file or IPython.
    """
    _, graph = build_graph(node, tree=tree)
    if filename:
        return view(filename, graph)
    elif ipython:
        return browser(graph)

def build_graph(node, graph=None, context=None, tree=False):
    top = pydot.Node( id(node), label=node.name )

    if not graph:
        graph = pydot.Graph(graph_type='digraph')
        graph.add_node( top )
        context = Counter()
        context[node.name] += 1

    # Increment the name of the top node

    cluster = pydot.Cluster(str(id(node.children)), label=' ')

    for child in node.children:
        nd, _ = build_graph(child, graph, context, tree=True)

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

        nd = pydot.Node(id(nd), label=name)
        cluster.add_node(nd)

        graph.add_node( nd )
        graph.add_edge( pydot.Edge(top, nd) )

    graph.add_subgraph(cluster)

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
