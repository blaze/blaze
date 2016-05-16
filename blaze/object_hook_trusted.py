from __future__ import absolute_import, division, print_function

from .object_hook import object_hook, register
import werkzeug.exceptions as wz_ex
from .compatibility import builtins, reduce
import numpy as np
import pandas as pd

_converters_trusted = object_hook._converters.copy()


def object_hook_trusted(ob, _converters=_converters_trusted):
    return object_hook(ob, _converters=_converters)
object_hook_trusted._converters = _converters_trusted
object_hook_trusted.register = register(converters=_converters_trusted)
del _converters_trusted


@object_hook_trusted.register('builtin_function')
def builtins_function_from_str(f):
    if f in ("eval", "exec"):
        raise wz_ex.Forbidden("cannot invoke eval or exec")
    return getattr(builtins, f)


@object_hook_trusted.register('numpy_pandas_function')
def numpy_pandas_function_from_str(f):
    """
    reconstruct function from string representation
    """
    if f.startswith("numpy"):
        mod = np
    elif f.startswith("pandas"):
        mod = pd
    else:
        raise wz_ex.NotImplemented("accepts numpy/pandas/builtin funcs only")
    fcn = reduce(getattr, f.split('.')[1:], mod)
    return fcn
