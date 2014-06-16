from __future__ import absolute_import, division, print_function

from dynd import nd
from datashape.dispatch import dispatch
from datashape import DataShape
from datashape.user import validate, issubschema
import numpy as np


__all__ = ['into']


@dispatch((list, tuple, set), (list, tuple, set))
def into(a, b):
    return type(a)(b)

@dispatch(dict, (list, tuple, set))
def into(a, b):
    return dict(b)

@dispatch((list, tuple, set), dict)
def into(a, b):
    return type(a)(map(type(a), sorted(b.items(), key=lambda x: x[0])))

@dispatch(nd.array, object)
def into(a, b):
    return nd.array(b)

@dispatch(list, nd.array)
def into(a, b):
    return nd.as_py(b)

@dispatch(np.ndarray, nd.array)
def into(a, b):
    return nd.as_numpy(b, allow_copy=True)

@dispatch(np.ndarray, object)
def into(a, b):
    return np.asarray(b)

@dispatch(list, np.ndarray)
def into(a, b):
    return b.tolist()
