from __future__ import absolute_import, division, print_function

from .compatibility import basestring
from .dispatch import dispatch

import sqlalchemy as sa


__all__ = ()


@dispatch(sa.Table, basestring)
def create_index(s, column, name=None, unique=False):
    if name is None:
        raise ValueError('SQL indexes must have a name')
    sa.Index(name, s.c[column], unique=unique).create(s.bind)


@dispatch(sa.Table, list)
def create_index(s, columns, name=None, unique=False):
    if name is None:
        raise ValueError('SQL indexes must have a name')
    sa.Index(name, *(s.c[c] for c in columns), unique=unique).create(s.bind)
