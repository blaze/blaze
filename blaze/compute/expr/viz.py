"""
Visualize expression graphs using graphviz.
"""

try:
    import networkx
    have_networkx = True
except ImportError:
    have_networkx = False

from io import BytesIO
import warnings
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile

def dump(node, ipython=True):
    """
    Dump the expression graph to either a file or IPython.
    """
    if not networkx:
        warnings.warn("networkx not installed, unable to view graph")
        return

    graph = build_graph(networkx.DiGraph(), node, set())
    if ipython:
        return browser(graph)
    else:
        return view(graph)

def build_graph(graph, term, seen):
    if term in seen:
        return

    seen.add(term)
    for arg in term.args:
        graph.add_edge(term, arg)
        build_graph(graph, arg, seen)

    return graph

def browser(graph):
    from IPython.core.display import Image
    import networkx

    with NamedTemporaryFile(delete=True) as tempdot:
        networkx.write_dot(graph, tempdot.name)
        tempdot.flush()
        p = Popen(['dot', '-Tpng', tempdot.name], stdout=PIPE)
        pngdata = BytesIO(p.communicate()[0]).read()

    return Image(data=pngdata)

def view(self):
    import matplotlib.pyplot as plt
    networkx.draw(self)
    plt.show()
