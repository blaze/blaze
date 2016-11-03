from __future__ import absolute_import, division, print_function

from ..dispatch import dispatch
from .core import compute_up, optimize
from ..expr import (
    Field,
    BinOp,
    Expr)
from distributed import client as dist
from dask import delayed

class DataFrame:
    """A proxy referring to a remote DataFrame.
    'data' may be a `future` or a `delayed`."""
    def __init__(self, data):
        self.data = data
class Series:
    """A proxy referring to a remote Series.
    'data' may be a `future` or a `delayed`."""
    def __init__(self, data):
        self.data = data

@dispatch(Expr, (Series, DataFrame))
def post_compute(expr, result, scope=None):
    """ Effects after the computation is complete """
    c = dist.default_client()
    f, = c.compute([result.data])
    return f

#
# All these compute_up functions merely accumulate computation functions into a `Task` object,
# which then is submitted in the `post_compute` call.
#

@dispatch(Field, DataFrame)
def compute_up(t, df, **kwargs):
    assert len(t.fields) == 1
    return Series(delayed(lambda x:x[t.fields[0]])(df.data))


@dispatch(Field, Series)
def compute_up(t, s, **kwargs):
    assert len(t.fields) == 1
    return Series(delayed(lambda x:x)(s.data))


@dispatch(BinOp, Series)
def compute_up(t, s, **kwargs):
    """Binary op where one of the two operands is a literal, such as `a + 1`."""
    if isinstance(t.lhs, Expr):
        return Series(delayed(lambda x,y:x.op(y, x.rhs))(t, s.data))
    else:
        return Series(delayed(lambda x,y:x.op(x.lhs, y))(t, s.data))


@compute_up.register(BinOp, Series, Series)
def compute_up_binop(t, lhs, rhs, **kwargs):
    return Series(delayed(lambda x,y,z:x.op(y,z))(t, lhs.data, rhs.data))
