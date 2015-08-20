from __future__ import absolute_import, division, print_function

from datetime import datetime, date, timedelta
import sys


def identity(x):
    return x


def asday(dt):
    if isinstance(dt, datetime):
        return dt.date()
    else:
        return dt


def asweek(dt):
    if isinstance(dt, datetime):
        dt = dt.date()
    return dt - timedelta(days=dt.isoweekday() - 1)


def ashour(dt):
    return datetime(dt.year, dt.month, dt.day, dt.hour, tzinfo=dt.tzinfo)


def asminute(dt):
    return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                    tzinfo=dt.tzinfo)


def assecond(dt):
    return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                    dt.second, tzinfo=dt.tzinfo)


def asmillisecond(dt):
    return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                    dt.second, dt.microsecond // 1000, tzinfo=dt.tzinfo)

if sys.version_info < (2, 7):
    def total_seconds(td):
        """ Total seconds of a timedelta

        For Python 2.6 compatibility
        """
        return (td.microseconds + 1e6 * (td.seconds + 24 * 3600 * td.days)) / 1e6
else:
    total_seconds = timedelta.total_seconds


unit_map = {'year': 'asyear',
            'month': 'asmonth',
            'week': 'asweek',
            'day': 'asday',
            'hour': 'ashour',
            'minute': 'asminute',
            'second': 'assecond',
            'millisecond': 'asmillisecond',
            'microsecond': identity}


def truncate_year(dt, measure):
    """
    Truncate by years

    >>> dt = datetime(2003, 6, 25, 12, 30, 0)
    >>> truncate_year(dt, 1)
    datetime.date(2003, 1, 1)

    >>> truncate_year(dt, 5)
    datetime.date(2000, 1, 1)
    """
    return date(dt.year // measure * measure, 1, 1)


def truncate_month(dt, measure):
    """
    Truncate by months

    >>> dt = datetime(2000, 10, 25, 12, 30, 0)
    >>> truncate_month(dt, 1)
    datetime.date(2000, 10, 1)

    >>> truncate_month(dt, 4)
    datetime.date(2000, 8, 1)
    """
    months = dt.year * 12 + dt.month
    months = months // measure * measure
    return date((months - 1) // 12, (months - 1) % 12 + 1, 1)


def truncate_day(dt, measure):
    """
    Truncate by days

    >>> dt = datetime(2000, 6, 27, 12, 30, 0)
    >>> truncate_day(dt, 1)
    datetime.date(2000, 6, 27)
    >>> truncate_day(dt, 3)
    datetime.date(2000, 6, 25)

    """
    days = dt.toordinal()
    days = days // measure * measure
    return date.fromordinal(days)

oneday = timedelta(days=1)


def truncate_week(dt, measure):
    """
    Truncate by weeks

    >>> dt = datetime(2000, 6, 22, 12, 30, 0)
    >>> truncate_week(dt, 1)
    datetime.date(2000, 6, 18)
    >>> truncate_week(dt, 3)
    datetime.date(2000, 6, 4)

    Weeks are defined by having isoweekday == 7 (Sunday)
    >>> truncate_week(dt, 1).isoweekday()
    7
    """
    return truncate_day(dt, measure * 7)


epoch = datetime.utcfromtimestamp(0)


def utctotimestamp(dt):
    """
    Convert a timestamp to seconds

    >>> dt = datetime(2000, 1, 1)
    >>> utctotimestamp(dt)
    946684800.0

    >>> datetime.utcfromtimestamp(946684800)
    datetime.datetime(2000, 1, 1, 0, 0)
    """
    return total_seconds(dt - epoch)


def truncate_minute(dt, measure):
    """
    Truncate by minute

    >>> dt = datetime(2000, 1, 1, 12, 30, 38)
    >>> truncate_minute(dt, 1)
    datetime.datetime(2000, 1, 1, 12, 30)
    >>> truncate_minute(dt, 12)
    datetime.datetime(2000, 1, 1, 12, 24)
    """
    return asminute(truncate_second(dt, measure * 60))


def truncate_hour(dt, measure):
    """
    Truncate by hour

    >>> dt = datetime(2000, 1, 1, 12, 30, 38)
    >>> truncate_hour(dt, 1)
    datetime.datetime(2000, 1, 1, 12, 0)
    >>> truncate_hour(dt, 5)
    datetime.datetime(2000, 1, 1, 10, 0)
    """
    return ashour(truncate_second(dt, measure * 3600))


def truncate_second(dt, measure):
    """
    Truncate by second

    >>> dt = datetime(2000, 1, 1, 12, 30, 38)
    >>> truncate_second(dt, 15)
    datetime.datetime(2000, 1, 1, 12, 30, 30)
    """
    d = datetime(
        dt.year, dt.month, dt.day, tzinfo=dt.tzinfo)  # local zero for seconds
    seconds = total_seconds(dt - d) // measure * measure
    return dt.utcfromtimestamp(seconds + utctotimestamp(d))


def truncate_millisecond(dt, measure):
    """
    Truncate by millisecond

    >>> dt = datetime(2000, 1, 1, 12, 30, 38, 12345)
    >>> truncate_millisecond(dt, 5)
    datetime.datetime(2000, 1, 1, 12, 30, 38, 10000)
    """
    d = datetime(
        dt.year, dt.month, dt.day, tzinfo=dt.tzinfo)  # local zero for seconds
    seconds = total_seconds(dt - d) * 1000 // measure * measure / 1000. + 1e-7
    return dt.utcfromtimestamp(seconds + utctotimestamp(d))


def truncate_microsecond(dt, measure):
    """
    Truncate by microsecond

    >>> dt = datetime(2000, 1, 1, 12, 30, 38, 12345)
    >>> truncate_microsecond(dt, 100)
    datetime.datetime(2000, 1, 1, 12, 30, 38, 12300)
    """
    d = datetime(
        dt.year, dt.month, dt.day, tzinfo=dt.tzinfo)  # local zero for seconds
    seconds = total_seconds(dt - d) * 1000000 // measure * measure / 1000000
    return dt.utcfromtimestamp(seconds + utctotimestamp(d))


truncate_functions = {'year': truncate_year,
                      'month': truncate_month,
                      'week': truncate_week,
                      'day': truncate_day,
                      'hour': truncate_hour,
                      'minute': truncate_minute,
                      'second': truncate_second,
                      'millisecond': truncate_millisecond,
                      'microsecond': truncate_microsecond}


def truncate(dt, measure, unit):
    """ Truncate datetimes

    Examples
    --------

    >>> dt = datetime(2003, 6, 25, 12, 30, 0)
    >>> truncate(dt, 1, 'day')
    datetime.date(2003, 6, 25)

    >>> truncate(dt, 5, 'hours')
    datetime.datetime(2003, 6, 25, 10, 0)

    >>> truncate(dt, 3, 'months')
    datetime.date(2003, 6, 1)
    """
    from blaze.expr.datetime import normalize_time_unit
    unit = normalize_time_unit(unit)
    return truncate_functions[unit](dt, measure)
