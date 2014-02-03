"""
Blaze expression graph for deferred evaluation. Each expression node has
an opcode and operands. An operand is a Constant or another expression node.
Each expression node carries a DataShape as type.
"""

from __future__ import absolute_import, division, print_function

from functools import partial

#------------------------------------------------------------------------
# Opcodes
#------------------------------------------------------------------------

array = 'array'    # array input
kernel = 'kernel'  # kernel application, carrying the blaze kernel as a
                   # first argument (Constant)

#------------------------------------------------------------------------
# Graph
#------------------------------------------------------------------------

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

# ______________________________________________________________________
# Graph constructors

ArrayOp    = partial(Op, array)

# Kernel application. Associated metadata:
#   kernel: the blaze.function.Kernel that was applied
#   overload: the blaze.overload.Overload that selected for the input args

KernelOp   = partial(Op, kernel)
