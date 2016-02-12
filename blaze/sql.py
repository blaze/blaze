from __future__ import absolute_import, division, print_function

import sqlalchemy as sa

from .compatibility import basestring
from .dispatch import dispatch

__all__ = ()


# sqlite raises OperationalError but postgres raises ProgrammingError
# thanks fam
_errors = sa.exc.ProgrammingError, sa.exc.OperationalError


@dispatch(sa.Table, basestring)
def create_index(s, column, name=None, unique=False, ignore_existing=False):
    if name is None:
        raise ValueError('SQL indexes must have a name')
    try:
        sa.Index(name, s.c[column], unique=unique).create(s.bind)
    except _errors:
        if not ignore_existing:
            raise


@dispatch(sa.Table, (list, tuple))
def create_index(s, columns, name=None, unique=False, ignore_existing=False):
    if name is None:
        raise ValueError('SQL indexes must have a name')
    try:
        sa.Index(
            name,
            *(s.c[c] for c in columns),
            unique=unique
        ).create(s.bind)
    except _errors:
        if not ignore_existing:
            raise
