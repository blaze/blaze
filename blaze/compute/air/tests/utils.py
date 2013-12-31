from __future__ import absolute_import, division, print_function

from datashape import dshape
import blaze
from blaze.compute.ops.ufuncs import add, mul
from blaze.compute.air import from_expr, ExecutionContext

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def make_graph():
    a = blaze.array(range(10), dshape('10, int32'))
    b = blaze.array(range(10), dshape('10, float64'))
    c = blaze.array([i+0j for i in range(10)],
                    dshape('10, complex128'))

    result = mul(add(a, b), c)
    graph, expr_ctx = result.expr

    ctx = ExecutionContext()
    f, values = from_expr(graph, expr_ctx, ctx)

    return f, values, graph
