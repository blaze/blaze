from blaze.compute.pydatetime import truncate
from datetime import datetime, date

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
