"""
Deferred data descriptor for deferred expressions. This is backed up by an
actual deferred expression graph.
"""

from __future__ import absolute_import, division, print_function

import blaze

from . import I_DDesc, Capabilities

#------------------------------------------------------------------------
# Decorators
#------------------------------------------------------------------------

def force_evaluation(methname):
    """
    Wrap a method and make it force evaluation when called and dispatch the
    call to the resulting Array.
    """
    def method(self, *args, **kwargs):
        result = blaze.eval(blaze.Array(self))
        self._result = result
        method = getattr(result._data, methname)
        return method(*args, **kwargs)

    return method

#------------------------------------------------------------------------
# Data Descriptor
#------------------------------------------------------------------------

# TODO: Re-purpose this to work for general deferred computations, not just
#       those backed up by the expression graph of Blaze kernels

class DeferredDescriptor(I_DDesc):
    """
    Data descriptor for arrays backed up by a deferred expression graph.

    Attributes:
    -----------
    dshape: DataShape
        Intermediate type resolved as far as it can be typed over the
        sub-expressions

    expr  : (Op, ExprContext)
        The expression graph along with the expression context, see blaze.expr
    """

    def __init__(self, dshape, expr):
        self._dshape = dshape
        self.expr = expr

        # Result of evaluation (cached)
        self._result = None

    @property
    def inputs(self):
        graph, ctx = self.expr
        return [term for term in ctx.terms.values()
                         if isinstance(term, blaze.Array)]

    @property
    def dshape(self):
        return self._dshape

    @property
    def capabilities(self):
        """The capabilities for the deferred data descriptor."""
        return Capabilities(
            immutable = True,
            deferred = True,
            # persistency is not supported yet
            persistent = False,
            appendable = False,
            remote = False,
            )

    __array__           = force_evaluation('__array__')
    __iter__            = force_evaluation('__iter__')
    __getitem__         = force_evaluation('__getitem__')
    __len__             = force_evaluation('__len__')
