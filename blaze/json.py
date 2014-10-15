from __future__ import absolute_import, division, print_function

import json
from toolz import map, partial
import gzip

from .resource import resource

__all__ = 'resource',

@resource.register('.*\.json')
def resource_json(uri, open=open):
    f = open(uri)
    try:
        data = json.load(f)
        f.close()
        return data
    except ValueError:
        f = open(uri)
        data = map(json.loads, f)
        return data


@resource.register('.*\.json.gz')
def resource_json_gzip(uri):
    return resource_json(uri, open=partial(gzip.open, mode='rt'))
