from __future__ import absolute_import, division, print_function


class Excel(object):
    __slots__ = 'path', 'worksheet'
    def __init__(self, path, worksheet=None):
        self.path = path
        if worksheet is not None:
            raise NotImplementedError("Only supports first worksheet for now")
        self.worksheet = 0

from ..api.resource import resource
@resource.register('.*\.(xls|xlsx)')
def resource_excel(path, **kwargs):
    return Excel(path, **kwargs)
