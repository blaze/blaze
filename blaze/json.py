from __future__ import absolute_import, division, print_function

import json
from toolz import map, partial, concat
import gzip

from into import resource

__all__ = 'resource',

@resource.register('.*\.json')
def resource_json(uri, open=open, **kwargs):
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
def resource_json_gzip(uri, encoding='utf-8', **kwargs):
    f = gzip.open(uri)
    lines = (line.decode(encoding) for line in gzip.open(uri))
    try:
        one = json.loads(next(lines))
        two = json.loads(next(lines))
    except StopIteration:  # single json element
        f.close()
        return one
    except ValueError:  # Single multi-line element
        f.close()
        f = gzip.open(uri)
        o = json.loads(f.read().decode(encoding))
        f.close()
        return o
    # JSON Streaming case
    return concat([[one, two], map(json.loads, lines)])
    f.close()
