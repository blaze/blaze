from __future__ import absolute_import, division, print_function

from functools import partial
import numpy as np
import tables as tb
from blaze.expr import (Selection, Head, Column, ColumnWise, Projection,
                        TableSymbol, Sort, Reduction, count, Sample)
from blaze.expr import eval_str
from blaze.compatibility import basestring, map
from datashape import Record
from ..dispatch import dispatch


@dispatch(tb.Table)
def discover(t):
    return t.shape[0] * Record([[col, t.coltypes[col]] for col in t.colnames])


@dispatch(tb.Table)
def drop(t):
    t.remove()


@dispatch(tb.Table, basestring)
def create_index(t, column, name=None, **kwargs):
    create_index(getattr(t.cols, column), **kwargs)


@dispatch(tb.Table, list)
def create_index(t, columns, name=None, **kwargs):
    if not all(map(partial(hasattr, t.cols), columns)):
        raise ValueError('table %s does not have all passed in columns %s' %
                         (t, columns))
    for column in columns:
        create_index(t, column, **kwargs)


@dispatch(tb.Column)
def create_index(c, optlevel=9, kind='full', name=None, **kwargs):
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

@dispatch(Sample, tb.Table)
def compute_one(expr, data, **kwargs):
    """ A small,randomly selected sample of data from the given tables.Table

    Parameters
    ----------
    expr : TableExpr
        The TableExpr that we are calculating over
    data : tables.Table
        The pytables Table we are sampling from

    Returns
    -------
    numpy.ndarray
        A new numpy.ndarray (Yes, a numpy.ndarray not a tb.Table)

    Notes
    -----
    Each time compute(expression.sample(), tables.Table) is called a new, different
    numpy.ndarray is returned.

    Example
    -------
    >>> x = np.array([(1, 'Alice', 100),
              (2, 'Bob', -200),
              (3, 'Charlie', 300),
              (4, 'Denis', 400),
              (5, 'Edith', -500)],
             dtype=[('id', '<i8'), ('name', 'S7'), ('amount', '<i8')])
    >>> import tempfile
    >>> f = tempfile.mktemp('.h5', mode="w")
    >>> d = f.create_table('/', 'pytablestest',  x)
    >>> t = TableSymbol('t', '{id: int, name: string, amount: int}')
    >>> result = compute(t.sample(2), d)
    >>> assert(len(result) == 2)
    """

    array_len = len(data)
    count = expr.n
    if count > array_len and expr.replacement is False:
        #If we make it here, the user has requested more values than can be returned
        #  So, we need to pare things down.
        #In essence, this now works like a permutation()
        count = array_len

    indices=np.random.choice(array_len, count, replace = expr.replacement)
    result=data[indices]

    return result
    
