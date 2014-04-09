"""
Deferred data descriptor for deferred expressions. This is backed up by an
actual deferred expression graph.
"""

from __future__ import absolute_import, division, print_function

import blaze

from . import DDesc, Capabilities

#------------------------------------------------------------------------
# Data Descriptor
#------------------------------------------------------------------------

class Stream_DDesc(DDesc):
    """
    Data descriptor for arrays exposing mainly an iterator interface.

    Attributes:
    -----------
    dshape: DataShape
        Intermediate type resolved as far as it can be typed over the
        sub-expressions

    expr: (Op, ExprContext) or string
        The expression graph along with the expression context, or a
        string defining the expression.
    """

    def __init__(self, iterator, dshape, expr):
        self._iterator = iterator
        self._dshape = dshape
        self.expr = expr

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

    def __getitem__(self, key):
        """Streams do not support random seeks.
        """
        raise NotImplementedError

    def __iter__(self):
        return self._iterator
