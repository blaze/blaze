from __future__ import absolute_import, division, print_function

import types
from collections import Callable

from blaze.dispatch import dispatch
from functools import partial, wraps

from .json_dumps import json_dumps


json_dumps_trusted_ns = dict()
dispatch = partial(dispatch, namespace=json_dumps_trusted_ns)


@dispatch(Callable)
def json_dumps_trusted(f):
    # let the server serialize any callable - this is only used for testing
    # at present - do the error handling when json comes from client so in
    # object_hook, catch anything that is not pandas_numpy
    fcn = ".".join([f.__module__, f.__name__])
    return {'__!callable': fcn}


for types, func in json_dumps.funcs.items():
    json_dumps_trusted.register(types)(func)
