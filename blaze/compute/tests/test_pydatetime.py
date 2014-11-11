from blaze.compute.pydatetime import truncate
from datetime import datetime, date, timedelta

def test_hour():
    dts = [datetime(2000, 6, 20,  1, 00, 00),
           datetime(2000, 6, 20, 12, 59, 59),
           datetime(2000, 6, 20, 12, 00, 00),
           datetime(2000, 6, 20, 11, 59, 59)]

    assert [truncate(dt, 1, 'hour') for dt in dts] == \
            [datetime(2000, 6, 20,  1, 0),
             datetime(2000, 6, 20, 12, 0),
             datetime(2000, 6, 20, 12, 0),
             datetime(2000, 6, 20, 11, 0)]


def test_month():
    dts = [datetime(2000, 7, 1),
           datetime(2000, 6, 30),
           datetime(2000, 6, 1),
           datetime(2000, 5, 31)]

    assert [truncate(dt, 1, 'month') for dt in dts] == \
            [date(2000, 7, 1),
             date(2000, 6, 1),
             date(2000, 6, 1),
             date(2000, 5, 1),]

    assert truncate(datetime(2000, 12, 1), 1, 'month') == \
            date(2000, 12, 1)


def test_week():
    d = date(2014, 11, 8)
    assert truncate(d, 1, 'week').isoweekday() == 7
    assert (d - truncate(d, 1, 'week')) < timedelta(days=7)
    assert (d - truncate(d, 1, 'week')) > timedelta(days=0)
