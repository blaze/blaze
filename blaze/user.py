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
