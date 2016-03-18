from __future__ import absolute_import, division, print_function

import sqlalchemy as sa

from .compatibility import basestring
from .dispatch import dispatch

__all__ = ()


@dispatch(sa.Table, basestring)
def create_index(s, column, **kwargs):
    """Create an index for a single column.

    Parameters
    ----------
    s : sa.Table
        The table to create the index on.
    column : str
        The name of the column to create an index on.
    name : str
        The name of the created index.
    unique : bool, optional
        Should the index be unique.
    ignore_existing : bool, optional
        Should this supress exceptions from the index already existing.
    concurrently : bool, optional
        Should the index be created without holding a lock. This feature is
        postgres specific.
    """
    return create_index(s, (column,), **kwargs)


@dispatch(sa.Table, (list, tuple))
def create_index(s,
                 columns,
                 name=None,
                 unique=False,
                 ignore_existing=False,
                 concurrently=False):
    """Create an index for a single column.

    Parameters
    ----------
    s : sa.Table
        The table to create the index on.
    columns : list[str] or tuple[str]
        The names of the columns to create an index on.
    name : str
        The name of the created index.
    unique : bool, optional
        Should the index be unique.
    ignore_existing : bool, optional
        Should this supress exceptions from the index already existing.
    concurrently : bool, optional
        Should the index be created without holding a lock. This feature is
        postgres specific.
    """

    if name is None:
        raise ValueError('SQL indexes must have a name')
    try:
        sa.Index(
            name,
            *(s.c[c] for c in columns),
            unique=unique,
            **{'postgresql_concurrently': concurrently} if concurrently else {}
        ).create(s.bind)
    except (sa.exc.ProgrammingError, sa.exc.OperationalError):
        # sqlite raises OperationalError but postgres raises ProgrammingError
        # thanks fam
        if not ignore_existing:
            raise
