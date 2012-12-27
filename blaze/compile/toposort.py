from functools import partial
from itertools import ifilter

from collections import Counter

#------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------

OP  = 0
APP = 1
VAL = 2
FUN = 3

#------------------------------------------------------------------------
# Graph Sorting
#------------------------------------------------------------------------

def khan_sort(pred, graph):
    """
    See: Kahn, Arthur B. (1962), "Topological sorting of large networks"
    """
    result = []
    count = Counter()

    for node in graph:
        for child in iter(node):
            count[child] += 1

    sort = [node for node in graph if not count[node]]

    while sort:
        node = sort.pop()
        result.append(node)

        for child in iter(node):
            count[child] -= 1
            if count[child] == 0:
                sort.append(child)

    result.reverse()

    # Collect all the nodes thats satisfy the selecter property.
    # For example, all the value nodes or all the op nodes.
    return list(ifilter(pred, result))

def tarjan_sort(pred, graph):
    raise NotImplementedError

def toposort(pred, graph, algorithm='khan'):
    """
    Sort the expression graph topologically to resolve the order needed
    to execute operations.
    """

    #
    #     +
    #    / \
    #   a   +     --> [a, b, c, d]
    #      / \
    #     b   c
    #         |
    #         d
    #

    if algorithm == 'khan':
        return khan_sort(pred, graph)
    if algorithm == 'tarjan':
        return tarjan_sort(pred, graph)
    else:
        raise NotImplementedError

#------------------------------------------------------------------------
# Sorters
#------------------------------------------------------------------------

topovals = partial(toposort, lambda x: x.kind == VAL)
topops   = partial(toposort, lambda x: x.kind == OP)
topfuns  = partial(toposort, lambda x: x.kind == FUN)

#------------------------------------------------------------------------
# Functors
#------------------------------------------------------------------------

#          fmap f
#    a              f(a)
#   / \              / \
#  b   c     =    f(b) f(c)
#     /  \             /  \
#    d    e         f(d)   f(e)

def fmap(f, tree):
    """ Functor for trees """
    # this is trivial because all nodes use __slots__
    x1 = copy(tree)
    for x in tree:
        x1.children = fmap(f, x1.children)
    return x1
