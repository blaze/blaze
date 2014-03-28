"""
Blaze expression graph for deferred evaluation. Each expression node has
an opcode and operands. An operand is a Constant or another expression node.
Each expression node carries a DataShape as type.
"""

from __future__ import absolute_import, division, print_function
from functools import partial


class BlazeExprNode(object):
    """Abstract base class for blaze expressions"""

    def __new__(cls, *args, **kwargs):
        """Use new to automatically return canonicalized classes"""
        return

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        super(BlazeExprNode, self).__init__()


class OpNode(BlazeExprNode):
    """Abstract node representing the node is an operator"""
    pass


class ParallelOpNode(BlazeExprNode):
    """Abstract node representing the node is a parallel operator"""
    pass


class ArithmeticNode(BlazeExprNode):
    """Node for simple arithmetic operators"""
    pass


class MutatorNode(BlazeExprNode):
    """Abstract node representing shape changing qualities of a node"""
    pass


class AppendNode(MutatorNode):
    """Node for appending datasets together"""
    pass


class MapNode(ParallelOpNode):
    """Node for a mapper operation"""
    pass


class ReduceNode(ParallelOpNode):
    """Node for a reducer operation"""
    pass


class SelectorNode(BlaseExprNode):
    """Abstract node representing the node is an selection"""
    pass


class GroupByNode(BlazeExprNode):
    """Node for groupby selection"""
    pass


class SliceNode(SelectorNode):
    """Node for slicing operations"""
    pass


class IndexNode(SelectorNode):
    """Node for indexing into calculation"""
    pass


##############################################################################
## Deprecated
##############################################################################
array = 'array'    # array input
kernel = 'kernel'  # kernel application, carrying the blaze kernel as a
                   # first argument (Constant)


class ExprContext(object):
    """
    Context for blaze graph expressions.

    This keeps track of a mapping between graph expression nodes and the
    concrete data inputs (i.e. blaze Arrays).

    Attributes:
    ===========

    terms: { ArrayOp: Array }
        Mapping from ArrayOp nodes to inputs
    """

    def __init__(self, contexts=[]):
        # Coercion constraints between types with free variables
        self.constraints = []
        self.terms = {} # All terms in the graph, { Array : Op }
        self.params = []

        for ctx in contexts:
            self.constraints.extend(ctx.constraints)
            self.terms.update(ctx.terms)
            self.params.extend(ctx.params)

    def add_input(self, term, data):
        if term not in self.terms:
            self.params.append(term)
        self.terms[term] = data


class Op(object):
    """
    Single node in blaze expression graph.

    Attributes
    ----------
    opcode: string
        Kind of the operation, i.e. 'array' or 'kernel'

    uses: [Op]
        Consumers (or parents) of this result. This is useful to keep
        track of, since we always start evaluation from the 'root', and we
        may miss tracking outside uses. However, for correct behaviour, these
        need to be retained
    """

    def __init__(self, opcode, dshape, *args, **metadata):
        self.opcode = opcode
        self.dshape = dshape
        self.uses = []
        self.args = list(args)

        if opcode == 'kernel':
            assert 'kernel' in metadata
            assert 'overload' in metadata
        self.metadata = metadata

        for arg in self.args:
            arg.add_use(self)

    def add_use(self, use):
        self.uses.append(use)

    def __repr__(self):
        opcode = self.opcode
        if opcode == kernel:
            opcode = self.metadata["kernel"]
        metadata = ", ".join(self.metadata)
        return "%s(...){dshape(%s), %s}" % (opcode, self.dshape, metadata)

    def tostring(self):
        subtrees = " -+- ".join(map(str, self.args))
        node = str(self)
        length = max(len(subtrees), len(node))
        return "%s\n%s" % (node.center(len(subtrees) / 2), subtrees.center(length))


ArrayOp = partial(Op, array)

# Kernel application. Associated metadata:
#   kernel: the blaze.function.Kernel that was applied
#   overload: the blaze.overload.Overload that selected for the input args
KernelOp = partial(Op, kernel)
