from __future__ import absolute_import, division, print_function

import warnings

import datashape
from datashape import String, DataShape, Option, bool_
from odo.utils import copydoc

from .expressions import schema_method_list, ElemWise
from .arithmetic import Interp, Repeat, _mkbin, repeat, interp, _add, _radd
from ..compatibility import basestring

__all__ = ['Like',
           'like',
           'strlen',
           'str_len',
           'str_upper',
           'str_lower',
           'str_cat',
           'StrCat',
           'UnaryStringFunction']


class Like(ElemWise):

    """ Filter expression by string comparison

    >>> from blaze import symbol, like, compute
    >>> t = symbol('t', 'var * {name: string, city: string}')
    >>> expr = t[t.name.like('Alice*')]

    >>> data = [('Alice Smith', 'New York'),
    ...         ('Bob Jones', 'Chicago'),
    ...         ('Alice Walker', 'LA')]
    >>> list(compute(expr, data))
    [('Alice Smith', 'New York'), ('Alice Walker', 'LA')]
    """
    __slots__ = '_hash', '_child', 'pattern'

    def _dshape(self):
        shape, schema = self._child.dshape.shape, self._child.schema
        schema = Option(bool_) if isinstance(schema.measure, Option) else bool_
        return DataShape(*(shape + (schema,)))


@copydoc(Like)
def like(child, pattern):
    if not isinstance(pattern, basestring):
        raise TypeError('pattern argument must be a string')
    return Like(child, pattern)


class UnaryStringFunction(ElemWise):

    """String function that only takes a single argument.
    """
    __slots__ = '_hash', '_child'


class strlen(UnaryStringFunction):

    def __init__(self, *args, **kwargs):
        warnings.warn("`strlen()` has been deprecated in 0.10 and will be "
                      "removed in 0.11.  Use ``str_len()`` instead.",
                      DeprecationWarning)
        super(self, strlen).__init__(*args, **kwargs)


class str_len(UnaryStringFunction):
    schema = datashape.int64


class str_upper(UnaryStringFunction):

    @property
    def schema(self):
        return self._child.schema


class str_lower(UnaryStringFunction):

    @property
    def schema(self):
        return self._child.schema


class StrCat(ElemWise):
    """
    Concatenate two string columns together with optional 'sep' argument.

    >>> from blaze.expr import symbol
    >>> from datashape import dshape

    >>> ds = dshape('3 * {name: string[10], comment: string[25], num: int32}')
    >>> s = symbol('s', dshape=ds)
    >>> data = [('alice', 'this is good', 0),
                ('suri', 'this is not good', 1),
                ('jinka', 'this is ok', 2)]
    >>> df = pd.DataFrame(data, columns=['name', 'comment', 'num'])

    >>> compute(s.name.str_cat(s.comment, sep=' -- '), df)
        0       alice -- this is good
        1    suri -- this is not good
        2         jinka -- this is ok
        Name: name, dtype: object


    Invoking str_cat() on a non string column raises a TypeError during compute

    >>> compute(s.name.str_cat(s.num, sep=' -- '), df)
    TypeError: can only concat string columns
    """
    __slots__ = '_hash', 'lhs', 'rhs', 'sep'
    __inputs__ = 'lhs', 'rhs'

    def _dshape(self):
        '''
        since pandas supports concat for string columns, do the same for blaze
        '''
        new_s_len = \
            self.lhs.schema.measure.fixlen + self.rhs.schema.measure.fixlen
        shape, schema = self.lhs.dshape.shape, DataShape(String(new_s_len))
        return DataShape(*(shape + (schema,)))


@copydoc(StrCat)
def str_cat(expr, to_concat, sep=None):
    # pandas supports concat for string columns only, do the same for blaze
    if not isstring(to_concat.dshape):
        raise TypeError("can only concat string columns")

    if sep is not None:
        if not isinstance(sep, basestring):
            raise TypeError("keyword argument 'sep' must be a String")

    return StrCat(expr, to_concat, sep=sep)


def isstring(ds):
    measure = ds.measure
    return isinstance(getattr(measure, 'ty', measure), String)


_mod, _rmod = _mkbin('mod', Interp)
_mul, _rmul = _mkbin('mul', Repeat)


schema_method_list.extend([(isstring,
                            set([_add,
                                 _radd,
                                 _mod,
                                 _rmod,
                                 _mul,
                                 _rmul,
                                 repeat,
                                 interp,
                                 like,
                                 str_len,
                                 strlen,
                                 str_upper,
                                 str_lower,
                                 str_cat]))])
