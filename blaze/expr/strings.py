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
           'StrFind',
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
    _arguments = '_child', 'pattern'

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
    _arguments = '_child',


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


class StrFind(ElemWise):
    """
    Find literal substring in string column.

    """

    __slots__ = '_hash', '_child', 'sub'
    schema = datashape.Option(datashape.int64)


@copydoc(StrFind)
def str_find(col, sub):
    if not isinstance(sub, basestring):
        raise TypeError("'sub' argument must be a String")
    return StrFind(col, sub)



class StrCat(ElemWise):
    """
    Concatenate two string columns together with optional 'sep' argument.

    >>> import pandas as pd
    >>> from blaze import symbol, compute, dshape

    >>> ds = dshape('3 * {name: ?string, comment: ?string, num: int32}')
    >>> s = symbol('s', dshape=ds)
    >>> data = [('al', 'good', 0), ('suri', 'not good', 1), ('jinka', 'ok', 2)]
    >>> df = pd.DataFrame(data, columns=['name', 'comment', 'num'])

    >>> compute(s.name.str_cat(s.comment, sep=' -- '), df)
    0          al -- good
    1    suri -- not good
    2         jinka -- ok
    Name: name, dtype: object

    For rows with null entries, it returns null. This is consistent with
    default pandas behavior with kwarg: na_rep=None.

    >>> data = [(None, None, 0), ('suri', 'not good', 1), ('jinka', None, 2)]
    >>> df = pd.DataFrame(data, columns=['name', 'comment', 'num'])
    >>> compute(s.name.str_cat(s.comment, sep=' -- '), df)
    0                 NaN
    1    suri -- not good
    2                 NaN
    Name: name, dtype: object

    """
    _arguments = 'lhs', 'rhs', 'sep'
    __inputs__ = 'lhs', 'rhs'

    def _dshape(self):
        '''
        since pandas supports concat for string columns, do the same for blaze
        '''
        shape = self.lhs.dshape.shape
        if isinstance(self.lhs.schema.measure, Option):
            schema = self.lhs.schema
        elif isinstance(self.rhs.schema.measure, Option):
            schema = self.rhs.schema
        else:
            _, lhs_encoding = self.lhs.schema.measure.parameters
            _, rhs_encoding = self.rhs.schema.measure.parameters
            assert lhs_encoding == rhs_encoding
            # convert fixed length string to variable length string
            schema = DataShape(String(None, lhs_encoding))

        return DataShape(*(shape + (schema,)))


@copydoc(StrCat)
def str_cat(lhs, rhs, sep=None):
    """
    returns lhs + sep + rhs

    Raises:
        Invoking on a non string column raises a TypeError
        If kwarg 'sep' is not a string, raises a TypeError
    """
    # pandas supports concat for string columns only, do the same for blaze
    if not isstring(rhs.dshape):
        raise TypeError("can only concat string columns")

    if sep is not None:
        if not isinstance(sep, basestring):
            raise TypeError("keyword argument 'sep' must be a String")

    return StrCat(lhs, rhs, sep=sep)


def isstring(ds):
    measure = ds.measure
    return isinstance(getattr(measure, 'ty', measure), String)


_mod, _rmod = _mkbin('mod', Interp)
_mul, _rmul = _mkbin('mul', Repeat)


class str_ns(object):

    def __init__(self, field):
        self.field = field

    def upper(self):
        return str_upper(self.field)

    def lower(self):
        return str_lower(self.field)

    def len(self):
        return str_len(self.field)

    def like(self, pattern):
        return like(self.field, pattern)

    def cat(self, other, sep=None):
        return str_cat(self.field, other, sep=sep)

    def find(self, sub):
        return str_find(self.field, sub)


class str(object):

    __name__ = 'str'

    def __get__(self, obj, type=None):
        return str_ns(obj)


schema_method_list.extend([(isstring,
                            set([_add,
                                 _radd,
                                 _mod,
                                 _rmod,
                                 _mul,
                                 _rmul,
                                 str(),
                                 repeat,
                                 interp,
                                 like,
                                 str_len,
                                 strlen,
                                 str_upper,
                                 str_lower,
                                 str_cat]))])
