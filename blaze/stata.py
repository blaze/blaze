from __future__ import absolute_import, division, print_function

import pandas

from .api.resource import resource

__all__ = ['resource']


@resource.register('.*\.dta')
def resource_stata(filename, **kwargs):
    return pandas.read_stata(filename, **kwargs)
