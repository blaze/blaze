from __future__ import absolute_import, division, print_function

import blz
import numpy as np
from pandas import DataFrame

from .dispatch import dispatch
from .compute.blz import *


@dispatch((tuple, set, list, type, object), (blz.btable, blz.barray))
def into(o, b):
    return into(o, into(np.ndarray(0), b))


@dispatch(np.ndarray, (blz.btable, blz.barray))
def into(a, b):
    return b[:]


@dispatch(DataFrame, blz.btable)
def into(a, b, columns=None, schema=None):
    if not columns and schema:
        columns = dshape(schema)[0].names
    return DataFrame.from_items(((column, b[column][:]) for column in
                                    sorted(b.names)),
                                orient='columns',
                                columns=columns)
