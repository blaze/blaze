from __future__ import absolute_import, division, print_function

from ..dispatch import dispatch
from ..data.csv import CSV
from ..expr import Expr, Symbol
from ..utils import available_memory
import os

@dispatch(Expr, CSV)
def pre_compute(expr, data, **kwargs):
    if os.path.getsize(data.path) < available_memory() / 4:
        return data.pandas_read_csv()
    else:
        return data

