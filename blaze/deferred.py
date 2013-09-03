# -*- coding: utf-8 -*-

"""
Deferred array node.
"""

from __future__ import print_function, division, absolute_import
from blaze.expr import dump


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

    def __str__(self):
        term, ctx = self.expr
        return "Deferred(%s, %s)" % (self.dshape, term)

    def view(self):
        term, context = self.expr
        ipython = False
        try:
            ipython = __IPYTHON__
        except NameError:
            pass

        return dump(term, ipython=ipython)
