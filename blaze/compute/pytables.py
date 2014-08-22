from __future__ import absolute_import, division, print_function

import numpy as np
import tables as tb
from blaze.expr import (Selection, Head, Column, ColumnWise, Projection,
                        TableSymbol, Sort, Reduction, count)
from blaze.expr import eval_str
from blaze.compatibility import basestring
from datashape import Record
from ..dispatch import dispatch


@dispatch(tb.Table)
def discover(t):
    return t.shape[0] * Record([[col, t.coltypes[col]] for col in t.colnames])


@dispatch(tb.Table)
def drop(t):
    t.remove()


@dispatch(object)
def create_index(o):
    """Create an index on a column.

    Parameters
    ----------
    o : indexable object

    Examples
    --------
    >>> import tables as tb
    >>> import numpy as np
    >>> data = [(1, 2.0, 'a'), (2, 3.0, 'b'), (3, 4.0, 'c')]
    >>> arr = np.array(data, dtype=[('id', 'i8'), ('value', 'f8'),
    ...                             ('key', '|S1')])
    >>> f = tb.open_file(tempfile.mkstemp(), mode='w')
    >>> t = f.create_table('/', 'table', arr)
    >>> create_index(t, 'id')
    >>> 'id' in t.colsindexed
    >>> t.close()
    >>> f.close()
    """
    raise NotImplementedError("create_index not implemented for type %r" %
                              type(o).__name__)


@dispatch(tb.Table, basestring)
def create_index(t, column_name, **kwargs):
    create_index(getattr(t.cols, column_name), **kwargs)


@dispatch(tb.Table, (list, tuple))
def create_index(t, column_names, **kwargs):
    assert all(hasattr(t.cols, column_name) for column_name in column_names), \
        'table %s does not have all passed in columns %s' % (t, column_names)
    for column_name in column_names:
        create_index(t, column_name, **kwargs)


@dispatch(tb.Column)
def create_index(c, optlevel=9, kind='full', **kwargs):
    c.create_index(optlevel=optlevel, kind=kind, **kwargs)


@dispatch(Selection, tb.Table)
def compute_one(sel, t, **kwargs):
    s = eval_str(sel.predicate.expr)
    return t.read_where(s)


@dispatch(TableSymbol, tb.Table)
def compute_one(ts, t, **kwargs):
    return t


@dispatch(Reduction, (tb.Column, tb.Table))
def compute_one(r, c, **kwargs):
    return compute_one(r, c[:])


@dispatch(Projection, tb.Table)
def compute_one(proj, t, **kwargs):
    # only options here are
    # read the whole thing in and then select
    # or
    # create an output array that is at most the size of the on disk table and
    # fill it will the columns iteratively
    # both of these options aren't ideal but pytables has no way to select
    # multiple column subsets so pick the one where we can optimize for the best
    # case rather than prematurely pessimizing
    #
    # TODO: benchmark on big tables because i'm not sure exactly what the
    # implications here are for memory usage
    columns = proj.columns
    dtype = np.dtype([(col, t.dtype[col]) for col in columns])
    out = np.empty(t.shape, dtype=dtype)
    for c in columns:
        out[c] = t.col(c)
    return out


@dispatch(Column, tb.Table)
def compute_one(c, t, **kwargs):
    return getattr(t.cols, c.column)


@dispatch(count, tb.Column)
def compute_one(r, c, **kwargs):
    return len(c)


@dispatch(Head, tb.Table)
def compute_one(h, t, **kwargs):
    return t[:h.n]


@dispatch(ColumnWise, tb.Table)
def compute_one(c, t, **kwargs):
    uservars = dict((col, getattr(t.cols, col)) for col in c.active_columns())
    e = tb.Expr(str(c.expr), uservars=uservars, truediv=True)
    return e.eval()


@dispatch(Sort, tb.Table)
def compute_one(s, t, **kwargs):
    if isinstance(s.key, Column) and s.key.child.isidentical(s.child):
        key = s.key.name
    else:
        key = s.key
    assert hasattr(t.cols, key), 'Table has no column(s) %s' % s.key
    result = t.read_sorted(sortby=key, checkCSI=True)
    if s.ascending:
        return result
    return result[::-1]
