from __future__ import absolute_import, division, print_function

import blz
import numpy as np
from .dispatch import dispatch
from .compute.blz import *


@dispatch((type, object), blz.btable)
def into(o, b):
    return into(o, into(np.ndarray(0), b))


@dispatch(np.ndarray, blz.btable)
def into(a, b):
    return b[:]
