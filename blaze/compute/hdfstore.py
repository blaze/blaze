from __future__ import absolute_import, division, print_function

from collections import namedtuple

from .core import pre_compute
from ..expr import Expr, Field
from ..dispatch import dispatch

from odo import into, chunks

import pandas as pd


@dispatch(Expr, pd.io.pytables.AppendableFrameTable)
def pre_compute(expr, data, **kwargs):
    return into(chunks(pd.DataFrame), data, **kwargs)


@dispatch(Expr, pd.io.pytables.Fixed)
def pre_compute(expr, data, **kwargs):
    return into(pd.DataFrame, data, **kwargs)


@dispatch(Field, pd.HDFStore)
def compute_up(expr, data, **kwargs):
    key = '/' + expr._name
    if key in data.keys():
        return data.get_storer(key)
    else:
        return HDFGroup(data, key)


HDFGroup = namedtuple('HDFGroup', 'parent,datapath')


@dispatch(Field, HDFGroup)
def compute_up(expr, data, **kwargs):
    key = data.datapath + '/' + expr._name
    if key in data.parent.keys():
        return data.parent.get_storer(key)
    else:
        return HDFGroup(data.parent, key)
