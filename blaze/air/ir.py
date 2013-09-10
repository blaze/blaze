# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from pykit import types
from pykit.ir import Function, Builder, Value, Op, Const

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def qualified_name(f):
    return ".".join([f.__module__, f.__name__])

#------------------------------------------------------------------------
# AIR construction
#------------------------------------------------------------------------

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
    # Types

    argtypes = [operand.dshape for operand in inputs]
    signature = types.Function(graph.dshape, argtypes)

    # -------------------------------------------------
    # Setup function

    name = "expr%d" % ctx.incr()
    argnames = ["e%d" % i for i in range(len(inputs))]
    f = Function(name, argnames, signature)
    builder = Builder(f)
    builder.position_at_beginning(f.new_block('entry'))

    # -------------------------------------------------
    # Generate function

    values = dict((expr, f.get_arg("e%d" % i))
                      for i, expr in enumerate(inputs))
    _from_expr(graph, f, builder, values)

    retval = values[graph]
    builder.ret(retval)

    return f, values

def _from_expr(expr, f, builder, values):
    if expr.opcode == 'array':
        result = values[expr]
    else:
        # -------------------------------------------------
        # Construct args

        # This is purely for IR readability
        name = qualified_name(expr.metadata['func'])
        args = [_from_expr(arg, f, builder, values) for arg in expr.args]
        args = [Const(name)] + args

        # -------------------------------------------------
        # Construct Op

        result = Op("kernel", expr.dshape, args)
        result.add_metadata({
            'kernel': expr.metadata['kernel'],
            'func': expr.metadata['func'],
            'signature': expr.metadata['signature'],

        })
        builder.emit(result)

    values[expr] = result
    return result

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