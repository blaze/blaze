# -*- coding: utf-8 -*-

"""
Deferred array node and kernel application.
"""

from __future__ import print_function, division, absolute_import

from itertools import chain

import blaze
from blaze.datashape import coretypes as T
from blaze.expr import dump
from blaze.expr.context import merge

from blaze.py2help import dict_iteritems

#------------------------------------------------------------------------
# Deferred Array
#------------------------------------------------------------------------

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


#------------------------------------------------------------------------
# Application
#------------------------------------------------------------------------

def apply_kernel(kernel, *args, **kwargs):
    """
    Apply blaze kernel `kernel` to the given arguments.

    Returns: a Deferred node representation the delayed computation
    """
    from .expr import construct

    # -------------------------------------------------
    # Merge input contexts

    args, kwargs = blaze_args(args, kwargs)
    ctxs = collect_contexts(chain(args, kwargs.values()))
    ctx = merge(ctxs)

    # -------------------------------------------------
    # Find match to overloaded function

    match, args = kernel.dispatcher.lookup_dispatcher(args, kwargs,
                                                    ctx.constraints)

    # -------------------------------------------------
    # Construct graph

    term = construct.construct(kernel, ctx, match.func, match.dst_sig, args)
    return blaze.Deferred(term.dshape, (term, ctx))

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def blaze_args(args, kwargs):
    """Build blaze arrays from inputs to a blaze kernel"""
    args = [make_blaze(a) for a in args]
    kwargs = dict((v, make_blaze(k)) for k, v in dict_iteritems(kwargs))
    return args, kwargs

def make_blaze(value):
    if not isinstance(value, (blaze.Deferred, blaze.Array)):
        dshape = T.typeof(value)
        if not dshape.shape:
            value = [value]
        value = blaze.Array([value], dshape)
    return value

def collect_contexts(args):
    for term in args:
        if term.expr:
            t, ctx = term.expr
            yield ctx
