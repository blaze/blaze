# -*- coding: utf-8 -*-

"""
Deferred data descriptor for deferred expressions. This is backed up by an
actual deferred expression graph.
"""

from __future__ import print_function, division, absolute_import

import blaze
from .data_descriptor import IDataDescriptor

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

class DeferredDescriptor(IDataDescriptor):
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
        return [term for term in ctx.terms.itervalues()
                         if isinstance(term, blaze.Array)]

    @property
    def strategy(self):
        strategies = set([input._data.strategy for input in self.inputs])
        if len(strategies) > 1:
            raise ValueError(
                "Multiple execution strategies encounted: %s" % (strategies,))

        [strategy] = strategies
        return strategy

    @property
    def dshape(self):
        return self._dshape

    @property
    def immutable(self):
        """Returns True, a deferred blaze function is not concrete."""
        return True

    @property
    def deferred(self):
        """Returns True, blaze function arrays are deferred."""
        return True

    @property
    def persistent(self):
        # TODO: Maybe in the future we can support this, but not yet.
        return False

    @property
    def appendable(self):
        """A deferred function is not appendable."""
        return False

    __array__           = force_evaluation('__array__')
    __iter__            = force_evaluation('__iter__')
    __getitem__         = force_evaluation('__getitem__')
    __len__             = force_evaluation('__len__')
