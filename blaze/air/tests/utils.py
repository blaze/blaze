# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import blaze
from blaze.ops.ufuncs import add, mul
from blaze.air import from_expr, ExecutionContext

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def make_graph():
    a = blaze.array(range(10))
    b = blaze.array([float(x) for x in range(10)])
    c = blaze.array([complex(x, x) for x in range(10)])

    result = mul(add(a, b), c)
    graph, expr_ctx = result.expr

    ctx = ExecutionContext()
    f, values = from_expr(graph, expr_ctx, ctx)

    return f, values, graph