# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from pykit import types
from pykit.ir import Function, Builder, Value, Op

from blaze.py2help import dict_iteritems

def from_expr(graph, expr_context, ctx):
    """
    Map a Blaze expression graph to blaze AIR

    Parameters
    ----------
    graph: blaze.expr.Op
        Expression graph

    expr_context: ExprContext
        Context of the expression

    ctx: ExecutionContext
    """
    inputs = expr_context.inputs

    # -------------------------------------------------
    # Setup function

    name = "expr%d" % ctx.incr()
    argnames = ["e%d" % i for i in range(len(inputs))]
    signature = types.Function(graph.dshape, [types.Opaque] * len(inputs))
    f = Function(name, argnames, signature)
    builder = Builder(f)

    # -------------------------------------------------

    values = {} # term -> arg #
    _from_expr(graph, f, builder, values)
    return f, values


def _from_expr(expr, f, builder, values):
    if expr.opcode == 'array':
        result = values.get(expr) or f.get_arg("e%d" % len(values))
    else:
        result = Op("kernel", types.Opaque, [_from_expr(arg, f, builder, values)
                                             for arg in expr.args])
        builder.emit(result)

    values[expr] = result
    return result

def build_args(inputs):
    """
    Given a dict of inputs (term -> Array), return a map mapping original
    terms to unique terms according to the data inputs.
    """
    terms = {}
    arrays = {}
    for term, array in dict_iteritems(inputs):
        arrays.setdefault(array, term)
        terms[term] = arrays[term]

    return terms

#------------------------------------------------------------------------
# Execution context
#------------------------------------------------------------------------

class ExecutionContext(object):
    def __init__(self):
        self.count = 0

    def incr(self):
        count = self.count
        self.count += 1
        return count