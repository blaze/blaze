"""Array printing function

"""

from __future__ import absolute_import, division, print_function

from ...py2help import xrange

__all__ = ["array2string", "set_printoptions", "get_printoptions"]
__docformat__ = 'restructuredtext'


#
# Written by Konrad Hinsen <hinsenk@ere.umontreal.ca>
# last revision: 1996-3-13
# modified by Jim Hugunin 1997-3-3 for repr's and str's (and other details)
# and by Perry Greenfield 2000-4-1 for numarray
# and by Travis Oliphant  2005-8-22 for numpy
# and by Oscar Villellas 2013-4-30 for blaze
# and by Andy R. Terrel 2013-12-17 for blaze

import sys
# import numerictypes as _nt
# from umath import maximum, minimum, absolute, not_equal, isnan, isinf
import numpy as np
import numpy.core.umath as _um
import datashape
from datashape import Fixed, Var, TypeVar

from ...datadescriptor import IDataDescriptor, dd_as_py

# These are undesired dependencies:
from numpy import ravel, maximum, minimum, absolute

import inspect


def _dump_data_info(x, ident=None):
    ident = (ident if ident is not None
             else inspect.currentframe().f_back.f_lineno)
    if isinstance(x, IDataDescriptor):
        subclass = 'DATA DESCRIPTOR'
    elif isinstance(x, np.ndarray):
        subclass = 'NUMPY ARRAY'
    else:
        subclass = 'UNKNOWN'

    print('-> %s: %s: %s' % (ident, subclass, repr(x)))


def product(x, y):
    return x*y


def isnan(x):
    # hacks to remove when isnan/isinf are available for data descriptors
    if isinstance(x, IDataDescriptor):
        return _um.isnan(dd_as_py(x))
    else:
        return _um.isnan(x)


def isinf(x):
    if isinstance(x, IDataDescriptor):
        return _um.isinf(dd_as_py(x))
    else:
        return _um.isinf(x)


def not_equal(x, val):
    if isinstance(x, IDataDescriptor):
        return _um.not_equal(dd_as_py(x))
    else:
        return _um.not_equal(x, val)

# repr N leading and trailing items of each dimension
_summaryEdgeItems = 3

# total items > triggers array summarization
_summaryThreshold = 1000

_float_output_precision = 8
_float_output_suppress_small = False
_line_width = 75
_nan_str = 'nan'
_inf_str = 'inf'
_formatter = None  # formatting function for array elements

if sys.version_info[0] >= 3:
    from functools import reduce


def set_printoptions(precision=None, threshold=None, edgeitems=None,
                     linewidth=None, suppress=None,
                     nanstr=None, infstr=None,
                     formatter=None):
    """
    Set printing options.

    These options determine the way floating point numbers, arrays and
    other NumPy objects are displayed.

    Parameters
    ----------
    precision : int, optional
        Number of digits of precision for floating point output (default 8).
    threshold : int, optional
        Total number of array elements which trigger summarization
        rather than full repr (default 1000).
    edgeitems : int, optional
        Number of array items in summary at beginning and end of
        each dimension (default 3).
    linewidth : int, optional
        The number of characters per line for the purpose of inserting
        line breaks (default 75).
    suppress : bool, optional
        Whether or not suppress printing of small floating point values
        using scientific notation (default False).
    nanstr : str, optional
        String representation of floating point not-a-number (default nan).
    infstr : str, optional
        String representation of floating point infinity (default inf).
    formatter : dict of callables, optional
        If not None, the keys should indicate the type(s) that the respective
        formatting function applies to.  Callables should return a string.
        Types that are not specified (by their corresponding keys) are handled
        by the default formatters.  Individual types for which a formatter
        can be set are::

            - 'bool'
            - 'int'
            - 'float'
            - 'complexfloat'
            - 'longcomplexfloat' : composed of two 128-bit floats
            - 'numpy_str' : types `numpy.string_` and `numpy.unicode_`
            - 'str' : all other strings

        Other keys that can be used to set a group of types at once are::

            - 'all' : sets all types
            - 'int_kind' : sets 'int'
            - 'float_kind' : sets 'float'
            - 'complex_kind' : sets 'complexfloat'
            - 'str_kind' : sets 'str' and 'numpystr'

    See Also
    --------
    get_printoptions, set_string_function, array2string

    Notes
    -----
    `formatter` is always reset with a call to `set_printoptions`.

    Examples
    --------
    Floating point precision can be set:

    >>> np.set_printoptions(precision=4)
    >>> print(np.array([1.123456789]))
    [ 1.1235]

    Long arrays can be summarised:

    >>> np.set_printoptions(threshold=5)
    >>> print(np.arange(10))
    [0 1 2 ..., 7 8 9]

    Small results can be suppressed:

    >>> eps = np.finfo(float).eps
    >>> x = np.arange(4.)
    >>> x**2 - (x + eps)**2
    array([ -4.9304e-32,  -4.4409e-16,   0.0000e+00,   0.0000e+00])
    >>> np.set_printoptions(suppress=True)
    >>> x**2 - (x + eps)**2
    array([-0., -0.,  0.,  0.])

    A custom formatter can be used to display array elements as desired:

    >>> np.set_printoptions(formatter={'all':lambda x: 'int: '+str(-x)})
    >>> x = np.arange(3)
    >>> x
    array([int: 0, int: -1, int: -2])
    >>> np.set_printoptions()  # formatter gets reset
    >>> x
    array([0, 1, 2])

    To put back the default options, you can use:

    >>> np.set_printoptions(edgeitems=3,infstr='inf',
    ... linewidth=75, nanstr='nan', precision=8,
    ... suppress=False, threshold=1000, formatter=None)
    """

    global _summaryThreshold, _summaryEdgeItems, _float_output_precision
    global _line_width, _float_output_suppress_small, _nan_str, _inf_str
    global _formatter

    if linewidth is not None:
        _line_width = linewidth
    if threshold is not None:
        _summaryThreshold = threshold
    if edgeitems is not None:
        _summaryEdgeItems = edgeitems
    if precision is not None:
        _float_output_precision = precision
    if suppress is not None:
        _float_output_suppress_small = not not suppress
    if nanstr is not None:
        _nan_str = nanstr
    if infstr is not None:
        _inf_str = infstr
    _formatter = formatter


def get_printoptions():
    """
    Return the current print options.

    Returns
    -------
    print_opts : dict
        Dictionary of current print options with keys

          - precision : int
          - threshold : int
          - edgeitems : int
          - linewidth : int
          - suppress : bool
          - nanstr : str
          - infstr : str
          - formatter : dict of callables

        For a full description of these options, see `set_printoptions`.

    See Also
    --------
    set_printoptions, set_string_function

    """
    d = dict(precision=_float_output_precision,
             threshold=_summaryThreshold,
             edgeitems=_summaryEdgeItems,
             linewidth=_line_width,
             suppress=_float_output_suppress_small,
             nanstr=_nan_str,
             infstr=_inf_str,
             formatter=_formatter)
    return d


def _extract_summary(a):
    return l


def _leading_trailing(a):
    import numpy.core.numeric as _nc
    if len(a.dshape.shape) == 1:
        if len(a) > 2*_summaryEdgeItems:
            b = [dd_as_py(a[i]) for i in range(_summaryEdgeItems)]
            b.extend([dd_as_py(a[i]) for i in range(-_summaryEdgeItems, 0)])
        else:
            b = dd_as_py(a)
    else:
        if len(a) > 2*_summaryEdgeItems:
            b = [_leading_trailing(a[i])
                 for i in range(_summaryEdgeItems)]
            b.extend([_leading_trailing(a[-i])
                      for i in range(-_summaryEdgeItems, 0)])
        else:
            b = [_leading_trailing(a[i]) for i in range(0, len(a))]
    return b


def _boolFormatter(x):
    if x:
        return ' True'
    else:
        return 'False'


def repr_format(x):
    return repr(x)


def _apply_formatter(format_dict, formatter):
    fkeys = [k for k in formatter.keys() if formatter[k] is not None]
    if 'all' in fkeys:
        for key in formatdict.keys():
            formatdict[key] = formatter['all']
    if 'int_kind' in fkeys:
        for key in ['int']:
            formatdict[key] = formatter['int_kind']
    if 'float_kind' in fkeys:
        for key in ['float']:
            formatdict[key] = formatter['float_kind']
    if 'complex_kind' in fkeys:
        for key in ['complexfloat', 'longcomplexfloat']:
            formatdict[key] = formatter['complex_kind']
    if 'str_kind' in fkeys:
        for key in ['numpystr', 'str']:
            formatdict[key] = formatter['str_kind']
    for key in formatdict.keys():
        if key in fkeys:
            formatdict[key] = formatter[key]


def _choose_format(formatdict, ds):
    if isinstance(ds, datashape.DataShape):
        ds = ds[-1]

    if ds == datashape.bool_:
        format_function = formatdict['bool']
    elif ds in [datashape.int8, datashape.int16,
                datashape.int32, datashape.int64,
                datashape.uint8, datashape.uint16,
                datashape.uint32, datashape.uint64]:
        format_function = formatdict['int']
    elif ds in [datashape.float32, datashape.float64]:
        format_function = formatdict['float']
    elif ds in [datashape.complex_float32, datashape.complex_float64]:
        format_function = formatdict['complexfloat']
    elif isinstance(ds, datashape.String):
        format_function = formatdict['numpystr']
    else:
        format_function = formatdict['numpystr']

    return format_function


def _array2string(a, shape, dtype, max_line_width, precision,
                  suppress_small, separator=' ', prefix="", formatter=None):

    if any(isinstance(s, (Var, TypeVar)) for s in shape):
        dim_size = -1
    else:
        dim_size = reduce(product, shape, 1)

    if max_line_width is None:
        max_line_width = _line_width

    if precision is None:
        precision = _float_output_precision

    if suppress_small is None:
        suppress_small = _float_output_suppress_small

    if formatter is None:
        formatter = _formatter

    if dim_size > _summaryThreshold:
        summary_insert = "..., "
        data = ravel(np.array(_leading_trailing(a)))
    else:
        summary_insert = ""
        data = ravel(np.array(dd_as_py(a)))

    formatdict = {'bool': _boolFormatter,
                  'int': IntegerFormat(data),
                  'float': FloatFormat(data, precision, suppress_small),
                  'complexfloat': ComplexFormat(data, precision,
                                                suppress_small),
                  'numpystr': repr_format,
                  'str': str}

    if formatter is not None:
        _apply_formatter(formatdict, formatter)

    assert(not hasattr(a, '_format'))

    # find the right formatting function for the array
    format_function = _choose_format(formatdict, dtype)

    # skip over "["
    next_line_prefix = " "
    # skip over array(
    next_line_prefix += " "*len(prefix)

    lst = _formatArray(a, format_function, len(shape), max_line_width,
                       next_line_prefix, separator,
                       _summaryEdgeItems, summary_insert).rstrip()
    return lst


def _convert_arrays(obj):
    import numpy.core.numeric as _nc
    newtup = []
    for k in obj:
        if isinstance(k, _nc.ndarray):
            k = k.tolist()
        elif isinstance(k, tuple):
            k = _convert_arrays(k)
        newtup.append(k)
    return tuple(newtup)


def array2string(a, max_line_width=None, precision=None,
                 suppress_small=None, separator=' ', prefix="",
                 style=repr, formatter=None):
    """
    Return a string representation of an array.

    Parameters
    ----------
    a : ndarray
        Input array.
    max_line_width : int, optional
        The maximum number of columns the string should span. Newline
        characters splits the string appropriately after array elements.
    precision : int, optional
        Floating point precision. Default is the current printing
        precision (usually 8), which can be altered using `set_printoptions`.
    suppress_small : bool, optional
        Represent very small numbers as zero. A number is "very small" if it
        is smaller than the current printing precision.
    separator : str, optional
        Inserted between elements.
    prefix : str, optional
        An array is typically printed as::

          'prefix(' + array2string(a) + ')'

        The length of the prefix string is used to align the
        output correctly.
    style : function, optional
        A function that accepts an ndarray and returns a string.  Used only
        when the shape of `a` is equal to ``()``, i.e. for 0-D arrays.
    formatter : dict of callables, optional
        If not None, the keys should indicate the type(s) that the respective
        formatting function applies to.  Callables should return a string.
        Types that are not specified (by their corresponding keys) are handled
        by the default formatters.  Individual types for which a formatter
        can be set are::

            - 'bool'
            - 'int'
            - 'float'
            - 'complexfloat'
            - 'longcomplexfloat' : composed of two 128-bit floats
            - 'numpy_str' : types `numpy.string_` and `numpy.unicode_`
            - 'str' : all other strings

        Other keys that can be used to set a group of types at once are::

            - 'all' : sets all types
            - 'int_kind' : sets 'int'
            - 'float_kind' : sets 'float'
            - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
            - 'str_kind' : sets 'str' and 'numpystr'

    Returns
    -------
    array_str : str
        String representation of the array.

    Raises
    ------
    TypeError : if a callable in `formatter` does not return a string.

    See Also
    --------
    array_str, array_repr, set_printoptions, get_printoptions

    Notes
    -----
    If a formatter is specified for a certain type, the `precision` keyword is
    ignored for that type.

    Examples
    --------
    >>> x = np.array([1e-16,1,2,3])
    >>> print(np.array2string(x, precision=2, separator=',',
    ...                       suppress_small=True))
    [ 0., 1., 2., 3.]

    >>> x  = np.arange(3.)
    >>> np.array2string(x, formatter={'float_kind':lambda x: "%.2f" % x})
    '[0.00 1.00 2.00]'

    >>> x  = np.arange(3)
    >>> np.array2string(x, formatter={'int':lambda x: hex(x)})
    '[0x0L 0x1L 0x2L]'

    """
    shape, dtype = (a.dshape[:-1], a.dshape[-1])
    shape = tuple(int(x) if isinstance(x, Fixed) else x for x in shape)

    lst = _array2string(a, shape, dtype, max_line_width,
                        precision, suppress_small,
                        separator, prefix, formatter=formatter)
    return lst


def _extendLine(s, line, word, max_line_len, next_line_prefix):
    if len(line.rstrip()) + len(word.rstrip()) >= max_line_len:
        s += line.rstrip() + "\n"
        line = next_line_prefix
    line += word
    return s, line


def _formatArray(a, format_function, rank, max_line_len,
                 next_line_prefix, separator, edge_items, summary_insert):
    """formatArray is designed for two modes of operation:

    1. Full output

    2. Summarized output

    """
    if rank == 0:
        return format_function(dd_as_py(a)).strip()

    if summary_insert and 2*edge_items < len(a):
        leading_items = edge_items
        trailing_items = edge_items
        summary_insert1 = summary_insert
    else:
        leading_items, trailing_items, summary_insert1 = 0, len(a), ""

    if rank == 1:
        s = ""
        line = next_line_prefix
        for i in xrange(leading_items):
            word = format_function(dd_as_py(a[i])) + separator
            s, line = _extendLine(s, line, word, max_line_len,
                                  next_line_prefix)

        if summary_insert1:
            s, line = _extendLine(s, line, summary_insert1,
                                  max_line_len, next_line_prefix)

        for i in xrange(trailing_items, 1, -1):
            word = format_function(dd_as_py(a[-i])) + separator
            s, line = _extendLine(s, line, word, max_line_len,
                                  next_line_prefix)

        if len(a) > 0:
            word = format_function(dd_as_py(a[-1]))
            s, line = _extendLine(s, line, word, max_line_len, next_line_prefix)

        s += line + "]\n"
        s = '[' + s[len(next_line_prefix):]
    else:
        s = '['
        sep = separator.rstrip()
        for i in xrange(leading_items):
            if i > 0:
                s += next_line_prefix
            s += _formatArray(a[i], format_function, rank-1, max_line_len,
                              " " + next_line_prefix, separator, edge_items,
                              summary_insert)
            s = s.rstrip() + sep.rstrip() + '\n'*max(rank-1, 1)

        if summary_insert1:
            s += next_line_prefix + summary_insert1 + "\n"

        for i in xrange(trailing_items, 1, -1):
            if leading_items or i != trailing_items:
                s += next_line_prefix
            s += _formatArray(a[-i], format_function, rank-1, max_line_len,
                              " " + next_line_prefix, separator, edge_items,
                              summary_insert)
            s = s.rstrip() + sep.rstrip() + '\n'*max(rank-1, 1)
        if leading_items or trailing_items > 1:
            s += next_line_prefix
        s += _formatArray(a[-1], format_function, rank-1, max_line_len,
                          " " + next_line_prefix, separator, edge_items,
                          summary_insert).rstrip()+']\n'
    return s


class FloatFormat(object):
    def __init__(self, data, precision, suppress_small, sign=False):
        self.precision = precision
        self.suppress_small = suppress_small
        self.sign = sign
        self.exp_format = False
        self.large_exponent = False
        self.max_str_len = 0
        if data.dtype.kind in ['f', 'i', 'u']:
            self.fillFormat(data)

    def fillFormat(self, data):
        import numpy.core.numeric as _nc
        errstate = _nc.seterr(all='ignore')
        try:
            special = isnan(data) | isinf(data)
            valid = not_equal(data, 0) & ~special
            non_zero = absolute(data.compress(valid))
            if len(non_zero) == 0:
                max_val = 0.
                min_val = 0.
            else:
                max_val = maximum.reduce(non_zero)
                min_val = minimum.reduce(non_zero)
                if max_val >= 1.e8:
                    self.exp_format = True
                if not self.suppress_small and (min_val < 0.0001
                                                or max_val/min_val > 1000.):
                    self.exp_format = True
        finally:
            _nc.seterr(**errstate)

        if self.exp_format:
            self.large_exponent = 0 < min_val < 1e-99 or max_val >= 1e100
            self.max_str_len = 8 + self.precision
            if self.large_exponent:
                self.max_str_len += 1
            if self.sign:
                format = '%+'
            else:
                format = '%'
            format = format + '%d.%de' % (self.max_str_len, self.precision)
        else:
            format = '%%.%df' % (self.precision,)
            if len(non_zero):
                precision = max([_digits(x, self.precision, format)
                                 for x in non_zero])
            else:
                precision = 0
            precision = min(self.precision, precision)
            self.max_str_len = len(str(int(max_val))) + precision + 2
            if _nc.any(special):
                self.max_str_len = max(self.max_str_len,
                                       len(_nan_str),
                                       len(_inf_str)+1)
            if self.sign:
                format = '%#+'
            else:
                format = '%#'
            format = format + '%d.%df' % (self.max_str_len, precision)

        self.special_fmt = '%%%ds' % (self.max_str_len,)
        self.format = format

    def __call__(self, x, strip_zeros=True):
        import numpy.core.numeric as _nc
        err = _nc.seterr(invalid='ignore')

        try:
            if isnan(x):
                if self.sign:
                    return self.special_fmt % ('+' + _nan_str,)
                else:
                    return self.special_fmt % (_nan_str,)
            elif isinf(x):
                if x > 0:
                    if self.sign:
                        return self.special_fmt % ('+' + _inf_str,)
                    else:
                        return self.special_fmt % (_inf_str,)
                else:
                    return self.special_fmt % ('-' + _inf_str,)
        finally:
            _nc.seterr(**err)

        s = self.format % x
        if self.large_exponent:
            # 3-digit exponent
            expsign = s[-3]
            if expsign == '+' or expsign == '-':
                s = s[1:-2] + '0' + s[-2:]
        elif self.exp_format:
            # 2-digit exponent
            if s[-3] == '0':
                s = ' ' + s[:-3] + s[-2:]
        elif strip_zeros:
            z = s.rstrip('0')
            s = z + ' '*(len(s)-len(z))
        return s


def _digits(x, precision, format):
    s = format % x
    z = s.rstrip('0')
    return precision - len(s) + len(z)


if sys.version_info >= (3, 0):
    _MAXINT = 2**32 - 1
    _MININT = -2**32
else:
    _MAXINT = sys.maxint
    _MININT = -sys.maxint-1


class IntegerFormat(object):
    def __init__(self, data):
        try:
            max_str_len = max(len(str(maximum.reduce(data))),
                              len(str(minimum.reduce(data))))
            self.format = '%' + str(max_str_len) + 'd'
        except (TypeError, NotImplementedError):
            # if reduce(data) fails, this instance will not be called, just
            # instantiated in formatdict.
            pass
        except ValueError:
            # this occurs when everything is NA
            pass

    def __call__(self, x):
        if _MININT < x < _MAXINT:
            return self.format % x
        else:
            return "%s" % x


class ComplexFormat(object):
    def __init__(self, x, precision, suppress_small):
        self.real_format = FloatFormat(x.real, precision, suppress_small)
        self.imag_format = FloatFormat(x.imag, precision, suppress_small,
                                       sign=True)

    def __call__(self, x):
        r = self.real_format(x.real, strip_zeros=False)
        i = self.imag_format(x.imag, strip_zeros=False)
        if not self.imag_format.exp_format:
            z = i.rstrip('0')
            i = z + 'j' + ' '*(len(i)-len(z))
        else:
            i = i + 'j'
        return r + i


def _test():
    import blaze

    arr = blaze.array([2, 3, 4.0])
    print(arr.dshape)

    print(array2string(arr._data))

    arr = blaze.zeros('30, 30, 30, float32')
    print(arr.dshape)

    print(array2string(arr._data))
