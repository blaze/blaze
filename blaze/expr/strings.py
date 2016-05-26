from __future__ import absolute_import, division, print_function

import warnings

import datashape
from datashape import String, DataShape, Option, bool_
from odo.utils import copydoc

from .expressions import schema_method_list, ElemWise
from .arithmetic import Interp, Repeat, _mkbin, repeat, interp, _add, _radd
from ..compatibility import basestring, _inttypes

__all__ = ['Like',
           'like',
           'strlen',
           'str_len',
           'str_upper',
           'str_lower',
           'str_cat',
           'str_isalnum',
           'str_isalpha',
           'str_isdecimal',
           'str_isdigit',
           'str_islower',
           'str_isnumeric',
           'str_isspace',
           'str_istitle',
           'str_isupper',
           'StrCat',
           'StrFind',
           'StrSlice',
           'SliceReplace',
           'Replace',
           'Capitalize',
           'Strip',
           'LStrip',
           'RStrip',
           'Pad',
           'UnaryStringFunction']

def _validate(var, name, type, typename):
    if not isinstance(var, type):
        raise TypeError('"%s" argument must be a %s'%(name, typename))

def _validate_optional(var, name, type, typename):
    if var is not None and not isinstance(var, type):
        raise TypeError('"%s" argument must be a %s'%(name, typename))



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

class str_isalnum(UnaryStringFunction): schema = bool_
class str_isalpha(UnaryStringFunction): schema = bool_
class str_isdecimal(UnaryStringFunction): schema = bool_
class str_isdigit(UnaryStringFunction): schema = bool_
class str_islower(UnaryStringFunction): schema = bool_
class str_isnumeric(UnaryStringFunction): schema = bool_
class str_isspace(UnaryStringFunction): schema = bool_
class str_istitle(UnaryStringFunction): schema = bool_
class str_isupper(UnaryStringFunction): schema = bool_

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

class Replace(ElemWise):
    __slots__ = '_hash', '_child', 'old', 'new', 'max'
    schema = datashape.string

class Pad(ElemWise):
    __slots__ = '_hash', '_child', 'width', 'side', 'fillchar'
    schema = datashape.string

class Capitalize(UnaryStringFunction):
    schema = datashape.string

class Strip(UnaryStringFunction):
    schema = datashape.string

class LStrip(UnaryStringFunction):
    schema = datashape.string

class RStrip(UnaryStringFunction):
    schema = datashape.string

class StrSlice(ElemWise):
    __slots__ = '_hash', '_child', 'slice'
    schema = datashape.Option(datashape.string)

class SliceReplace(ElemWise):
    __slots__ = '_hash', '_child', 'start', 'stop', 'repl'
    schema = datashape.Option(datashape.string)

@copydoc(StrSlice)
def str_slice(col, idx):
    if not isinstance(idx, (slice, _inttypes)):
        raise TypeError("idx argument must be a slice or integer, given {}".format(slc))
    return StrSlice(col, (idx.start, idx.stop, idx.step) if isinstance(idx, slice) else idx)


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

    _validate_optional(sep, 'sep', basestring, 'string')
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

    def isalnum(self): return str_isalnum(self.field)
    def isalpha(self): return str_isalpha(self.field)
    def isdecimal(self): return str_isdecimal(self.field)
    def isdigit(self): return str_isdigit(self.field)
    def islower(self): return str_islower(self.field)
    def isnumeric(self): return str_isnumeric(self.field)
    def isspace(self): return str_isspace(self.field)
    def istitle(self): return str_istitle(self.field)
    def isupper(self): return str_isupper(self.field)

    def replace(self, old, new, max=None):
        _validate(old, 'old', basestring, 'string')
        _validate(new, 'new', basestring, 'string')
        _validate_optional(max, 'max', int, 'integer')
        return Replace(self.field, old, new, max)

    def capitalize(self):
        return Capitalize(self.field)

    def pad(self, width, side=None, fillchar=None):
        _validate(width, 'width', int, 'integer')
        if side not in (None, 'left', 'right'):
            raise TypeError('"side" argument must be either "left" or "right"')
        _validate_optional(fillchar, 'fillchar', basestring, 'string')
        return Pad(self.field, width, side, fillchar)

    def strip(self): return Strip(self.field)
    def lstrip(self): return LStrip(self.field)
    def rstrip(self): return RStrip(self.field)

    def __getitem__(self, idx):
        return str_slice(self.field, idx)

    def slice_replace(self, start=None, stop=None, repl=None):
        _validate_optional(start, 'start', int, 'integer')
        _validate_optional(stop, 'stop', int, 'integer')
        _validate_optional(repl, 'repl', basestring, 'string')
        return SliceReplace(self.field, start, stop, repl)

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
