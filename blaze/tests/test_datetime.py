from __future__ import absolute_import, division, print_function

import unittest
from datetime import date, time, datetime

import blaze
from datashape import dshape
from blaze.datadescriptor import ddesc_as_py


class TestDate(unittest.TestCase):
    def test_create(self):
        a = blaze.array(date(2000, 1, 1))
        self.assertEqual(a.dshape, dshape('date'))
        self.assertEqual(ddesc_as_py(a.ddesc), date(2000, 1, 1))
        a = blaze.array([date(1490, 3, 12), date(2020, 7, 15)])
        self.assertEqual(a.dshape, dshape('2 * date'))
        self.assertEqual(list(a), [date(1490, 3, 12), date(2020, 7, 15)])
        a = blaze.array(['1490-03-12', '2020-07-15'], dshape='date')
        self.assertEqual(a.dshape, dshape('2 * date'))
        self.assertEqual(list(a), [date(1490, 3, 12), date(2020, 7, 15)])

    def test_properties(self):
        a = blaze.array(['1490-03-12', '2020-07-15'], dshape='date')
        self.assertEqual(list(a.year), [1490, 2020])
        self.assertEqual(list(a.month), [3, 7])
        self.assertEqual(list(a.day), [12, 15])

class TestTime(unittest.TestCase):
    def test_create(self):
        a = blaze.array(time(14, 30))
        self.assertEqual(a.dshape, dshape('time'))
        self.assertEqual(ddesc_as_py(a.ddesc), time(14, 30))
        a = blaze.array([time(14, 30), time(12, 25, 39, 123456)])
        self.assertEqual(a.dshape, dshape('2 * time'))
        self.assertEqual(list(a), [time(14, 30), time(12, 25, 39, 123456)])
        a = blaze.array(['2:30 pm', '12:25:39.123456'], dshape='time')
        self.assertEqual(a.dshape, dshape('2 * time'))
        self.assertEqual(list(a), [time(14, 30), time(12, 25, 39, 123456)])

    def test_properties(self):
        a = blaze.array([time(14, 30), time(12, 25, 39, 123456)], dshape='time')
        self.assertEqual(list(a.hour), [14, 12])
        self.assertEqual(list(a.minute), [30, 25])
        self.assertEqual(list(a.second), [0, 39])
        self.assertEqual(list(a.microsecond), [0, 123456])

class TestDateTime(unittest.TestCase):
    def test_create(self):
        a = blaze.array(datetime(1490, 3, 12, 14, 30))
        self.assertEqual(a.dshape, dshape('datetime'))
        self.assertEqual(ddesc_as_py(a.ddesc), datetime(1490, 3, 12, 14, 30))
        a = blaze.array([datetime(1490, 3, 12, 14, 30),
                         datetime(2020, 7, 15, 12, 25, 39, 123456)])
        self.assertEqual(a.dshape, dshape('2 * datetime'))
        self.assertEqual(list(a), [datetime(1490, 3, 12, 14, 30),
                                   datetime(2020, 7, 15, 12, 25, 39, 123456)])
        a = blaze.array(['1490-mar-12 2:30 pm', '2020-07-15T12:25:39.123456'],
                        dshape='datetime')
        self.assertEqual(a.dshape, dshape('2 * datetime'))
        self.assertEqual(list(a), [datetime(1490, 3, 12, 14, 30),
                                   datetime(2020, 7, 15, 12, 25, 39, 123456)])

    def test_properties(self):
        a = blaze.array([datetime(1490, 3, 12, 14, 30),
                         datetime(2020, 7, 15, 12, 25, 39, 123456)],
                        dshape='datetime')
        self.assertEqual(list(a.date), [date(1490, 3, 12), date(2020, 7, 15)])
        self.assertEqual(list(a.time), [time(14, 30), time(12, 25, 39, 123456)])
        self.assertEqual(list(a.year), [1490, 2020])
        self.assertEqual(list(a.month), [3, 7])
        self.assertEqual(list(a.day), [12, 15])
        self.assertEqual(list(a.hour), [14, 12])
        self.assertEqual(list(a.minute), [30, 25])
        self.assertEqual(list(a.second), [0, 39])
        self.assertEqual(list(a.microsecond), [0, 123456])


if __name__ == '__main__':
    unittest.main()

