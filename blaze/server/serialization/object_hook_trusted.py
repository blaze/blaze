from __future__ import absolute_import, division, print_function

from .object_hook import object_hook, register
import werkzeug.exceptions as wz_ex
from blaze.compatibility import builtins, reduce
import numpy as np
import pandas as pd

_converters_trusted = object_hook._converters.copy()


def object_hook_trusted(ob, _converters=_converters_trusted):
    return object_hook(ob, _converters=_converters)
object_hook_trusted._converters = _converters_trusted
object_hook_trusted.register = register(converters=_converters_trusted)
del _converters_trusted


@object_hook_trusted.register('callable')
def numpy_pandas_function_from_str(f):
    """
    reconstruct function from string representation
    """
    if f.startswith(np.__name__):
        mod = np
    elif f.startswith(pd.__name__):
        mod = pd
    elif f.startswith(builtins.__name__):
        mod = builtins
    else:
        msg = ("Function {} not recognized; only numpy, pandas, or builtin "
               "functions are supported.")
        raise wz_ex.NotImplemented(msg.format(f))
    fcn = reduce(getattr, f.split('.')[1:], mod)
    return fcn
