from __future__ import absolute_import, division, print_function

from ..regex import RegexDispatcher

__all__ = 'resource',

resource = RegexDispatcher('resource')

@resource.register('.*', priority=1)
def resource_all(uri, *args, **kwargs):
    raise NotImplementedError("Unable to parse uri to data resource: " + uri)


@resource.register('.*::.*', priority=15)
def resource_split(uri, *args, **kwargs):
    uri, other = uri.rsplit('::', 1)
    return resource(uri, other, *args, **kwargs)
