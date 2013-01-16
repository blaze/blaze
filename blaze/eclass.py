#------------------------------------------------------------------------
# Evaluation Class ( eclass )
#------------------------------------------------------------------------

MANIFEST = 1
DELAYED  = 2

class eclass:
    manifest = MANIFEST
    delayed  = DELAYED

def all_manifest(list_of_nodes):
    from blaze.expr.nodes import Node
    return all(node.eclass == MANIFEST for node in list_of_nodes
            if isinstance(node, Node))

def all_deferred(list_of_nodes):
    return all(node.eclass == DELAYED for node in list_of_nodes)

def decide_eclass(a,b):
    if (a,b) == (MANIFEST, MANIFEST):
        return MANIFEST
    if (a,b) == (MANIFEST, DELAYED):
        return MANIFEST
    if (a,b) == (DELAYED, MANIFEST):
        return MANIFEST
    if (a,b) == (DELAYED, DELAYED):
        return DELAYED

def coerce_eclass(xs):
    return reduce(decide_eclass, (x.eclass for x in xs), DELAYED)
