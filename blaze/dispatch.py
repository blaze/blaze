from datashape.dispatch import namespace
from multipledispatch import dispatch
from functools import partial

dispatch = partial(dispatch, namespace=namespace)
