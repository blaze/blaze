from __future__ import absolute_import, division, print_function

import json

from .resource import resource
from toolz import map

__all__ = 'resource',

@resource.register('.*\.json')
def resource_json(uri):
    f = open(uri)
    try:
        data = json.load(f)
        f.close()
        return data
    except ValueError:
        f = open(uri)
        data = map(json.loads, f)
        return data
