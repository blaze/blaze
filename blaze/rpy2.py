from __future__ import absolute_import, division, print_function

from .dispatch import dispatch
from pandas import DataFrame
from rpy2.robjects import Sexp
from rpy2.robjects.pandas2ri import pandas2ri, ri2pandas

@dispatch(Sexp, DataFrame)
def into(_, b):
    return pandas2ri(b)

@dispatch(DataFrame, Sexp)
def into(_, b):
    return ri2pandas(b)
