# -*- coding: utf-8 -*-

"""
Translate blaze expressoin graphs into blaze AIR.
"""

from __future__ import absolute_import, division, print_function

import threading

from pykit import types
from pykit.ir import Function, Builder, Value, Op, Const

#------------------------------------------------------------------------
# AIR construction
#------------------------------------------------------------------------

def run(expr, env):
    graph, expr_ctx = expr
    air_func = from_expr(graph, expr_ctx)
    return air_func, env


def from_expr(graph, expr_context):
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
    inputs = expr_context.params

    # -------------------------------------------------
    # Types

    argtypes = [operand.dshape for operand in inputs]
    signature = types.Function(graph.dshape, argtypes, varargs=False)

    # -------------------------------------------------
    # Setup function

    name = "expr"
    argnames = ["e%d" % i for i in range(len(inputs))]
    f = Function(name, argnames, signature)
    builder = Builder(f)
    builder.position_at_beginning(f.new_block('entry'))

    # -------------------------------------------------
    # Generate function

    valuemap = dict((expr, f.get_arg("e%d" % i))
                      for i, expr in enumerate(inputs))
    _from_expr(graph, f, builder, valuemap)

    retval = valuemap[graph]
    builder.ret(retval)

    return f

def _from_expr(expr, f, builder, values):
    if expr.opcode == 'array':
        result = values[expr]
    else:
        # -------------------------------------------------
        # Construct args

        # This is purely for IR readability
        name = qualified_name(expr.metadata['overload'].func)
        args = [_from_expr(arg, f, builder, values) for arg in expr.args]
        args = [Const(name)] + args

        # -------------------------------------------------
        # Construct Op

        result = Op("kernel", expr.dshape, args)

        # Copy metadata verbatim
        assert 'kernel' in expr.metadata
        assert 'overload' in expr.metadata
        result.add_metadata(expr.metadata)

        # -------------------------------------------------
        # Emit Op in code stream

        builder.emit(result)

    values[expr] = result
    return result


#class ExecutionContext(object):
#    """Simple counter for variable names"""
#
#    # TODO: Why can't we just reuse expression names?
#
#    def __init__(self):
#        self.count = 0
#
#    def incr(self):
#        count = self.count
#        self.count += 1
#        return count
#
#
#_tls = threading.local()
#
#def current_ctx():
#    """Return the current evaluation strategy"""
#    try:
#        return _tls.ctx
#    except AttributeError:
#        _tls.ctx = ExecutionContext()
#        return _tls.ctx

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def qualified_name(f):
    return ".".join([f.__module__, f.__name__])
