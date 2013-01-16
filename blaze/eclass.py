#------------------------------------------------------------------------
# Evaluation Class ( eclass )
#------------------------------------------------------------------------

MANIFEST = 1
DELAYED  = 2

class eclass:
    """
    Enumeration of evaluation class::

        manifest | delayed
    """
    manifest = MANIFEST
    delayed  = DELAYED

def all_manifest(list_of_nodes):
    from blaze.expr.nodes import Node
    return all(node.eclass == MANIFEST for node in list_of_nodes
            if isinstance(node, Node))

def all_deferred(list_of_nodes):
    return all(node.eclass == DELAYED for node in list_of_nodes)

def _decide_eclass(a,b):
    """
    Decision procedure for deciding evaluation class of operands
    over 2 arguments.
    """

    if (a,b) == (MANIFEST, MANIFEST):
        return MANIFEST
    if (a,b) == (MANIFEST, DELAYED):
        return MANIFEST
    if (a,b) == (DELAYED, MANIFEST):
        return MANIFEST
    if (a,b) == (DELAYED, DELAYED):
        return DELAYED

def decide_eclass(xs):
    """
    Decision procedure for deciding evaluation class of operands
    over ``n`` arguments.
    """
    return reduce(decide_eclass, (x.eclass for x in xs), DELAYED)
