from __future__ import absolute_import, division, print_function

import numpy as np
import h5py

from ..expr import Reduction, Field, Projection, Broadcast, Selection
from ..expr import Distinct, Sort, Head, Label, ReLabel, Union, Expr, Slice
from ..expr import std, var, count, nunique
from ..expr import BinOp, UnaryOp, USub, Not

from .core import base, compute
from ..dispatch import dispatch
from ..api.into import into

__all__ = []


@dispatch(Slice, h5py.Dataset)
def compute_up(expr, data, **kwargs):
    return data[expr.index]
