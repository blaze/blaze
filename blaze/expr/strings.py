from __future__ import absolute_import, division, print_function

import warnings

import datashape
from datashape import String, DataShape, Option, bool_
from odo.utils import copydoc

from .expressions import schema_method_list, ElemWise
from .arithmetic import Interp, Repeat, _mkbin, repeat, interp, _add, _radd
from ..compatibility import basestring, _inttypes, builtins
from ..deprecation import deprecated

__all__ = ['Like',
           'like',
           'Pad',
           'Replace',
           'SliceReplace',
           # prevent 'len' to end up in global namespace
           #'len',
           'upper',
           'lower',
           'cat',
           'isalnum',
           'isalpha',
           'isdecimal',
           'isdigit',
           'islower',
           'isnumeric',
           'isspace',
           'istitle',
           'isupper',
           'StrCat',
           'find',
           'StrFind',
           'StrSlice',
           'slice',
           'slice_replace',
           'replace',
           'capitalize',
           'strip',
           'lstrip',
           'rstrip',
           'pad',
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


class len(UnaryStringFunction):
    schema = datashape.int64


class upper(UnaryStringFunction):

    @property
    def schema(self):
        return self._child.schema


class lower(UnaryStringFunction):

    @property
    def schema(self):
        return self._child.schema


class PredicateFunction(UnaryStringFunction):

    @property
    def schema(self):
        return bool_ if self._child.schema == datashape.string else Option(bool_)


class isalnum(PredicateFunction): pass
class isalpha(PredicateFunction): pass
class isdecimal(PredicateFunction): pass
class isdigit(PredicateFunction): pass
class islower(PredicateFunction): pass
class isnumeric(PredicateFunction): pass
class isspace(PredicateFunction): pass
class istitle(PredicateFunction): pass
class isupper(PredicateFunction): pass

class StrFind(ElemWise):
    """
    Find literal substring in string column.

    """

    _arguments = '_child', 'sub'
    schema = Option(datashape.int64)


@copydoc(StrFind)
def find(col, sub):
    if not isinstance(sub, basestring):
        raise TypeError("'sub' argument must be a String")
    return StrFind(col, sub)

class Replace(ElemWise):
    _arguments = '_child', 'old', 'new', 'max'

    @property
    def schema(self):
        return self._child.schema

def replace(col, old, new, max=None):
    _validate(old, 'old', basestring, 'string')
    _validate(new, 'new', basestring, 'string')
    _validate_optional(max, 'max', int, 'integer')
    return Replace(col, old, new, max)

class Pad(ElemWise):
    _arguments = '_child', 'width', 'side', 'fillchar'

    @property
    def schema(self):
        return self._child.schema

def pad(col, width, side=None, fillchar=None):
    _validate(width, 'width', int, 'integer')
    if side not in (None, 'left', 'right'):
        raise TypeError('"side" argument must be either "left" or "right"')
    _validate_optional(fillchar, 'fillchar', basestring, 'string')
    return Pad(col, width, side, fillchar)

class capitalize(UnaryStringFunction):

    @property
    def schema(self):
        return self._child.schema

class strip(UnaryStringFunction):

    @property
    def schema(self):
        return self._child.schema

class lstrip(UnaryStringFunction):

    @property
    def schema(self):
        return self._child.schema

class rstrip(UnaryStringFunction):

    @property
    def schema(self):
        return self._child.schema

class StrSlice(ElemWise):
    _arguments = '_child', 'slice'

    @property
    def schema(self):
        return self._child.schema

class SliceReplace(ElemWise):
    _arguments = '_child', 'start', 'stop', 'repl'

    @property
    def schema(self):
        return self._child.schema

def slice_replace(col, start=None, stop=None, repl=None):
    _validate_optional(start, 'start', int, 'integer')
    _validate_optional(stop, 'stop', int, 'integer')
    _validate_optional(repl, 'repl', basestring, 'string')
    return SliceReplace(col, start, stop, repl)


@copydoc(StrSlice)
def slice(col, idx):
    if not isinstance(idx, (builtins.slice, _inttypes)):
        raise TypeError("idx argument must be a slice or integer, given {}".format(slc))
    return StrSlice(col, (idx.start, idx.stop, idx.step) if isinstance(idx, builtins.slice) else idx)

class StrCat(ElemWise):
    """
    Concatenate two string columns together with optional 'sep' argument.

    >>> import pandas as pd
    >>> from blaze import symbol, compute, dshape

    >>> ds = dshape('3 * {name: ?string, comment: ?string, num: int32}')
    >>> s = symbol('s', dshape=ds)
    >>> data = [('al', 'good', 0), ('suri', 'not good', 1), ('jinka', 'ok', 2)]
    >>> df = pd.DataFrame(data, columns=['name', 'comment', 'num'])

    >>> compute(s.name.str.cat(s.comment, sep=' -- '), df)
    0          al -- good
    1    suri -- not good
    2         jinka -- ok
    Name: name, dtype: object

    For rows with null entries, it returns null. This is consistent with
    default pandas behavior with kwarg: na_rep=None.

    >>> data = [(None, None, 0), ('suri', 'not good', 1), ('jinka', None, 2)]
    >>> df = pd.DataFrame(data, columns=['name', 'comment', 'num'])
    >>> compute(s.name.str.cat(s.comment, sep=' -- '), df)
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
def cat(lhs, rhs, sep=None):
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

    def upper(self): return upper(self.field)
    def lower(self): return lower(self.field)
    def len(self): return len(self.field)
    def like(self, pattern): return like(self.field, pattern)
    def cat(self, other, sep=None): return cat(self.field, other, sep=sep)
    def find(self, sub): return find(self.field, sub)
    def isalnum(self): return isalnum(self.field)
    def isalpha(self): return isalpha(self.field)
    def isdecimal(self): return isdecimal(self.field)
    def isdigit(self): return isdigit(self.field)
    def islower(self): return islower(self.field)
    def isnumeric(self): return isnumeric(self.field)
    def isspace(self): return isspace(self.field)
    def istitle(self): return istitle(self.field)
    def isupper(self): return isupper(self.field)
    def replace(self, old, new, max=None): return replace(self.field, old, new, max)
    def capitalize(self): return capitalize(self.field)
    def pad(self, width, side=None, fillchar=None): return pad(self.field, width, side, fillchar)
    def strip(self): return strip(self.field)
    def lstrip(self): return lstrip(self.field)
    def rstrip(self): return rstrip(self.field)
    def __getitem__(self, idx): return slice(self.field, idx)
    def slice_replace(self, start=None, stop=None, repl=None):
        return slice_replace(self.field, start, stop, repl)

class str(object):

    __name__ = 'str'

    def __get__(self, obj, type):
        return str_ns(obj) if obj is not None else self


@deprecated('0.11', replacement='len()')
def str_len(*args, **kwds): return len(*args, **kwds)
@deprecated('0.11', replacement='upper()')
def str_upper(*args, **kwds): return upper(*args, **kwds)
@deprecated('0.11', replacement='lower()')
def str_lower(*args, **kwds): return lower(*args, **kwds)
@deprecated('0.11', replacement='cat(lhs, rhs, sep=None)')
def str_cat(*args, **kwds): return cat(*args, **kwds)


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
                                 str_len, # deprecated
                                 str_upper, # deprecated
                                 str_lower, # deprecated
                                 str_cat]))]) # deprecated
