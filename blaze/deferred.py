class Deferred(object):
    """
    Deferred Array node.

    Attributes:

        dshape: DataShape

            Intermediate type resolved as far as it can be typed over the
            sub-expressions

        expr  : (Op, ExprContext)

            The expression graph, see blaze.expr
    """

    def __init__(self, dshape, expr):
        self.dshape = dshape
        self.expr = expr