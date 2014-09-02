from __future__ import absolute_import, division, print_function

from ..regex import RegexDispatcher


resource = RegexDispatcher('resource')

@resource.register('.*', priority=1)
def resource_all(uri, *args, **kwargs):
    raise NotImplementedError("Unable to parse uri to data resource: " + uri)
