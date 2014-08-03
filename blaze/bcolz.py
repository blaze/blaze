from __future__ import absolute_import, division, print_function

import bcolz
import numpy as np
from pandas import DataFrame

from .dispatch import dispatch
from .compute.bcolz import *


@dispatch((type, object), (bcolz.ctable, bcolz.carray))
def into(o, b):
    return into(o, into(np.ndarray(0), b))


@dispatch(np.ndarray, (bcolz.ctable, bcolz.carray))
def into(a, b):
    return b[:]


@dispatch(DataFrame, bcolz.ctable)
def into(a, b, columns=None, schema=None):
    if not columns and schema:
        columns = dshape(schema)[0].names
    return DataFrame.from_items(((column, b[column][:]) for column in
                                    sorted(b.names)),
                                orient='columns',
                                columns=columns)
