from __future__ import absolute_import, division, print_function

from dynd import nd
from datashape.dispatch import dispatch
from datashape import DataShape
from datashape.user import validate, issubschema

from blaze import Array

@dispatch(DataShape, Array)
def validate(ds, arr):
    return issubschema(arr.dshape, ds)

@dispatch(DataShape, nd.array)
def validate(ds, arr):
    return issubschema(nd.dshape_of(arr), ds)

@dispatch((list, tuple, set), (list, tuple, set))
def into(a, b):
    return type(a)(b)

@dispatch(dict, (list, tuple, set))
def into(a, b):
    return dict(b)

@dispatch((list, tuple, set), dict)
def into(a, b):
    return type(a)(map(type(a), b.items()))

@dispatch(nd.array, object)
def into(a, b):
    return nd.array(b)

@dispatch(list, nd.array)
def into(a, b):
    return nd.as_py(b)

@dispatch((list, tuple, set), Array)
def into(a, b):
    if len(b.dshape.shape) == 1:
        return type(a)(b)
    else:
        return type(a)(into(a, item) for item in b)

@dispatch(nd.array, Array)
def into(a, b):
    return b.ddesc.dynd_arr()
