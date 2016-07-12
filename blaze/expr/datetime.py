from __future__ import absolute_import, division, print_function

from .expressions import ElemWise, schema_method_list, method_properties

import datashape
from datashape import dshape, isdatelike, isnumeric
from datashape.coretypes import timedelta_
from ..compatibility import basestring


__all__ = ['DateTime', 'Date', 'date', 'Year', 'year', 'Month', 'month', 'Day',
           'day', 'days', 'Hour', 'hour', 'Minute', 'minute', 'Second', 'second',
           'Millisecond', 'millisecond', 'Microsecond', 'microsecond', 'nanosecond',
           'Date',
           'date', 'Time', 'time', 'week', 'nanoseconds', 'seconds', 'total_seconds',
           'UTCFromTimestamp', 'DateTimeTruncate',
           'Ceil', 'Floor', 'Round', 'strftime']


def _validate(var, name, type, typename):
    if not isinstance(var, type):
        raise TypeError('"%s" argument must be a %s'%(name, typename))


class DateTime(ElemWise):
    """ Superclass for datetime accessors """
    _arguments = '_child',

    def __str__(self):
        return '%s.%s' % (str(self._child), type(self).__name__.lower())

    def _schema(self):
        ds = dshape(self._dtype)
        return ds if not isinstance(self._child.schema.measure, datashape.Option) else datashape.Option(ds)

    @property
    def _name(self):
        return '%s_%s' % (self._child._name, self.attr)

    @property
    def attr(self):
        return type(self).__name__.lower()


class Date(DateTime):
    _dtype = datashape.date_


def date(expr):
    return Date(expr)


class Year(DateTime):
    _dtype = datashape.int32


def year(expr):
    return Year(expr)


class Month(DateTime):
    _dtype = datashape.int64


def month(expr):
    return Month(expr)


class Day(DateTime):
    _dtype = datashape.int64


def day(expr):
    return Day(expr)


class Time(DateTime):
    _dtype = datashape.time_


def time(expr):
    return Time(expr)


class Hour(DateTime):
    _dtype = datashape.int64


def hour(expr):
    return Hour(expr)


class Minute(DateTime):
    _dtype = datashape.int64


def minute(expr):
    return Minute(expr)


class Second(DateTime):
    _dtype = datashape.int64


def second(expr):
    return Second(expr)


class Millisecond(DateTime):
    _dtype = datashape.int64


def millisecond(expr):
    return Millisecond(expr)


class Microsecond(DateTime):
    _dtype = datashape.int64


def microsecond(expr):
    return Microsecond(expr)


class UTCFromTimestamp(DateTime):
    _dtype = datashape.datetime_


def utcfromtimestamp(expr):
    return UTCFromTimestamp(expr)


class nanosecond(DateTime): _dtype = datashape.int64
class week(DateTime): _dtype = datashape.int64
class weekday(DateTime): _dtype = datashape.int64
class weekday_name(DateTime): _dtype = datashape.string
class daysinmonth(DateTime): _dtype = datashape.int64
class weekofyear(DateTime): _dtype = datashape.int64
class dayofyear(DateTime): _dtype = datashape.int64
class dayofweek(DateTime): _dtype = datashape.int64
class quarter(DateTime): _dtype = datashape.int64
class is_month_start(DateTime): _dtype = datashape.bool_
class is_month_end(DateTime): _dtype = datashape.bool_
class is_quarter_start(DateTime): _dtype = datashape.bool_
class is_quarter_end(DateTime): _dtype = datashape.bool_
class is_year_start(DateTime): _dtype = datashape.bool_
class is_year_end(DateTime): _dtype = datashape.bool_
class days_in_month(DateTime): _dtype = datashape.int64

class strftime(ElemWise):
    _arguments = '_child', 'format'
    schema = datashape.string

units = (
    'year',
    'month',
    'week',
    'day',
    'hour',
    'minute',
    'second',
    'millisecond',
    'microsecond',
    'nanosecond',
)


unit_aliases = {
    'y': 'year',
    'w': 'week',
    'd': 'day',
    'date': 'day',
    'h': 'hour',
    's': 'second',
    'ms': 'millisecond',
    'us': 'microsecond',
    'ns': 'nanosecond'
}


def normalize_time_unit(s):
    """ Normalize time input to one of 'year', 'second', 'millisecond', etc..

    Examples
    --------

    >>> normalize_time_unit('milliseconds')
    'millisecond'
    >>> normalize_time_unit('ms')
    'millisecond'
    """
    s = s.lower().strip()
    if s in units:
        return s
    if s in unit_aliases:
        return unit_aliases[s]
    if s[-1] == 's':
        return normalize_time_unit(s.rstrip('s'))

    raise ValueError("Do not understand time unit %s" % s)


class DateTimeTruncate(DateTime):
    _arguments = '_child', 'measure', 'unit'

    @property
    def _dtype(self):
        if units.index('day') >= units.index(self.unit):
            return datashape.date_
        else:
            return datashape.datetime_

    @property
    def _name(self):
        return self._child._name

    def __str__(self):
        return '%s.truncate(%ss=%g)' % (self._child, self.unit, self.measure)


def truncate(expr, *args, **kwargs):
    """ Truncate datetime expression

    Examples
    --------

    >>> from blaze import symbol, compute
    >>> from datetime import datetime
    >>> s = symbol('s', 'datetime')

    >>> expr = s.truncate(10, 'minutes')
    >>> compute(expr, datetime(2000, 6, 25, 12, 35, 10))
    datetime.datetime(2000, 6, 25, 12, 30)

    >>> expr = s.truncate(1, 'week')
    >>> compute(expr, datetime(2000, 6, 25, 12, 35, 10))
    datetime.date(2000, 6, 25)

    Alternatively use keyword arguments to specify unit and measure

    >>> expr = s.truncate(weeks=2)
    """
    if not args and not kwargs:
        raise TypeError('truncate takes exactly 2 positional arguments, '
                        'e.g., truncate(2, "days") or 1 keyword argument, '
                        'e.g., truncate(days=2)')
    if args:
        if kwargs:
            raise TypeError('Cannot pass both positional and keyword '
                            'arguments; given %s and %s.' % (args, kwargs))
        measure, unit = args
    else:
        [(unit, measure)] = kwargs.items()
    return DateTimeTruncate(expr, measure, normalize_time_unit(unit))


class UnaryDateTimeFunction(ElemWise):

    """DateTime function that only takes a single argument."""
    _arguments = '_child'


class Round(ElemWise):
    _arguments = '_child', 'freq'

    @property
    def schema(self):
        return self._child.schema


class Floor(ElemWise):
    _arguments = '_child', 'freq'

    @property
    def schema(self):
        return self._child.schema


class Ceil(ElemWise):
    _arguments = '_child', 'freq'

    @property
    def schema(self):
        return self._child.schema


class dt_ns(object):

    def __init__(self, field):
        self.field = field

    def year(self):
        return year(self.field)
    def month(self):
        return month(self.field)
    def day(self):
        return day(self.field)
    def hour(self):
        return hour(self.field)
    def minute(self):
        return minute(self.field)
    def date(self):
        return date(self.field)
    def time(self):
        return time(self.field)
    def second(self):
        return second(self.field)
    def millisecond(self):
        return millisecond(self.field)
    def microsecond(self):
        return microsecond(self.field)
    def nanosecond(self):
        return nanosecond(self.field)
    def weekday(self):
        return weekday(self.field)
    def weekday_name(self):
        return weekday_name(self.field)
    def daysinmonth(self):
        return daysinmonth(self.field)
    def weekofyear(self):
        return weekofyear(self.field)
    def dayofyear(self):
        return dayofyear(self.field)
    def dayofweek(self):
        return dayofweek(self.field)
    def quarter(self):
        return quarter(self.field)
    def is_month_start(self):
        return is_month_start(self.field)
    def is_month_end(self):
        return is_month_end(self.field)
    def is_quarter_start(self):
        return is_quarter_start(self.field)
    def is_quarter_end(self):
        return is_quarter_end(self.field)
    def is_year_start(self):
        return is_year_start(self.field)
    def is_year_end(self):
        return is_year_end(self.field)
    def days_in_month(self):
        return days_in_month(self.field)
    def strftime(self, format):
        _validate(format, 'format', basestring, 'string')
        return strftime(self.field, format)
    def truncate(self, *args, **kwargs):
        return truncate(self.field, *args, **kwargs)
    def round(self, freq):
        _validate(freq, 'freq', basestring, 'string')
        return Round(self.field, freq)
    def floor(self, freq):
        _validate(freq, 'freq', basestring, 'string')
        return Floor(self.field, freq)
    def ceil(self, freq):
        _validate(freq, 'freq', basestring, 'string')
        return Ceil(self.field, freq)
    def week(self):
        return week(self.field)

class dt(object):

    __name__ = 'dt'

    def __get__(self, obj, type=None):
        return dt_ns(obj) if obj is not None else self


class days(DateTime): _dtype = datashape.int64
class nanoseconds(DateTime): _dtype = datashape.int64
class seconds(DateTime): _dtype = datashape.int64
class total_seconds(DateTime): _dtype = datashape.int64


class timedelta_ns(object):

    def __init__(self, field):
        self.field = field

    def days(self): return days(self.field)
    def nanoseconds(self): return nanoseconds(self.field)
    def seconds(self): return seconds(self.field)
    def total_seconds(self): return total_seconds(self.field)


class timedelta(object):

    # pandas uses the same 'dt' name for
    # DateTimeProperties and TimedeltaProperties.
    __name__ = 'dt'

    def __get__(self, obj, type=None):
        return timedelta_ns(obj) if obj is not None else self


def isdeltalike(ds):
    return ds == timedelta_

schema_method_list.extend([
    (isdatelike, set([year, month, day, hour, minute, date, time, second,
                      millisecond, microsecond, truncate,
                      dt()])),
    (isnumeric, set([utcfromtimestamp])),
    (isdeltalike, set([timedelta()]))
])

method_properties |= set([year, month, day, hour, minute, second, millisecond,
                          microsecond, date, time, utcfromtimestamp])
