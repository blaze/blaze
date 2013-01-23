"""
This file contains stopgap solutions that should be nuked
whenever someone gets around to implementing something real.
"""

from blaze import error
from blaze.expr import graph, nodes, ops
from blaze.datashape import coretypes, DataShape

def _get_datashape(graph_node):
    if isinstance(graph_node, graph.App):
        return _get_datashape(graph_node.operator)
    elif isinstance(graph_node, nodes.Node):
        return graph_node.datashape
    else:
        return coretypes.from_python_scalar(graph_node)

def broadcast(*operands):
    types = [_get_datashape(op) for op in operands if op is not None]
    shapes = []
    for t in types:
        try:
            shapes.append(coretypes.extract_dims(t))
        except coretypes.NotNumpyCompatible:
            pass

    # TODO: broadcasting
    datashapes = [coretypes.to_numpy(coretypes.extract_measure(ds))
                      for ds in types]
    type = coretypes.promote_cvals(*datashapes)
    if not shapes:
        return type
    return DataShape(shapes[0] + (type,))

def compute_datashape(op, operands, kwargs):
    dshape = broadcast(*operands)

    if isinstance(op, ops.ReductionOp):
        if kwargs.get("axis", None):
            raise NotImplemented("axis")

        dshape = DataShape([coretypes.extract_measure(dshape)])

    return dshape
