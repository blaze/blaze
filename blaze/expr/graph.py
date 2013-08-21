"""
Blaze expression graph for deferred evaluation. Each expression node has
an opcode and operands. An operand is a Constant or another expression node.
Each expression node carries a DataShape as type.
"""

from functools import partial

#------------------------------------------------------------------------
# Opcodes
#------------------------------------------------------------------------

array = 'array'   # array input
const = 'const'   # constant value
kernel = 'kernel' # kernel application, carrying the blaze kernel as a
                  # first argument (Constant)

#------------------------------------------------------------------------
# Graph
#------------------------------------------------------------------------

class Op(object):
    """
    Single node in blaze expression graph.
    """

    def __init__(self, opcode, dshape, *args):
        self.opcode = opcode
        self.dshape = dshape
        self.args   = list(args)

# ______________________________________________________________________
# Graph constructors

ArrayOp    = partial(Op, array)
ConstantOp = partial(Op, const)
KernelOp   = partial(Op, kernel)