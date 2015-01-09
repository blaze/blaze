from __future__ import absolute_import, division, print_function

from .core import pre_compute
from ..expr import Expr, Field
from ..dispatch import dispatch
from into import into, chunks
import pandas as pd

@dispatch(Expr, pd.io.pytables.AppendableFrameTable)
def pre_compute(expr, data, **kwargs):
    return into(chunks(pd.DataFrame), data, **kwargs)


@dispatch(Expr, pd.io.pytables.Fixed)
def pre_compute(expr, data, **kwargs):
    return into(pd.DataFrame, data, **kwargs)


@dispatch(Field, pd.HDFStore)
def compute_up(expr, data, **kwargs):
    return data.get_storer('/%s' % expr._name)
