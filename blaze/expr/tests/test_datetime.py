from blaze.expr import symbol
import blaze.expr.datetime as bdt
from blaze.compatibility import builtins
from datashape import dshape

import pytest


def test_datetime_dshape():
    t = symbol('t', '5 * {name: string, when: datetime}')
    assert t.when.day.dshape == dshape('5 * int64')
    assert t.when.date.dshape == dshape('5 * date')


def test_date_attribute():
    t = symbol('t', '5 * {name: string, when: datetime}')
    expr = t.when.day
    assert eval(str(expr)).isidentical(expr)


def test_invalid_date_attribute():
    t = symbol('t', '5 * {name: string, when: datetime}')
    with pytest.raises(AttributeError):
        t.name.day


def test_date_attribute_completion():
    t = symbol('t', '5 * {name: string, when: datetime}')
    assert 'day' in dir(t.when)
    assert 'day' not in dir(t.name)
    assert not builtins.all([x.startswith('__') and x.endswith('__')
                            for x in dir(t.name)])


def test_datetime_attribute_name():
    t = symbol('t', '5 * {name: string, when: datetime}')
    assert 'when' in t.when.day._name


def test_utcfromtimestamp():
    t = symbol('t', '10 * {name: string, ts: int32}')
    assert t.ts.utcfromtimestamp.dshape == dshape('10 * datetime')


def test_isdatelike():
    assert not bdt.isdatelike('int32')
    assert bdt.isdatelike('?date')
    assert not bdt.isdatelike('{is_outdated: bool}')


def test_truncate_names():
    t = symbol('t', '5 * {name: string, when: datetime}')
    assert t.when.truncate(days=2)._name == 'when'


def test_truncate_repr():
    t = symbol('t', '5 * {name: string, when: datetime}')
    assert str(t.when.truncate(days=2)) == 't.when.truncate(days=2)'
    assert str(t.when.date.truncate(month=3)) == 't.when.date.truncate(months=3)'
    assert str(t.when.date.truncate(ns=4)) == 't.when.date.truncate(nanoseconds=4)'
    assert str(t.when.date.truncate(days=4.2)) == 't.when.date.truncate(days=4.2)'
    assert str(t.when.date.truncate(days=4.2876)) == 't.when.date.truncate(days=4.2876)'
    assert str(t.when.date.truncate(days=4.287654)) == 't.when.date.truncate(days=4.28765)'


def test_truncate_raises_with_no_arguments():
    t = symbol('t', '5 * {name: string, when: datetime}')
    with pytest.raises(TypeError):
        t.when.truncate()


@pytest.mark.parametrize('attr',
                         ['date', 'year', 'month', 'day', 'time', 'hour',
                          'second', 'millisecond', 'microsecond'])
def test_attributes(attr):
    t = symbol('t', 'var * datetime')
    assert getattr(t, attr).dshape is not None
    assert getattr(t, attr)._child is t


def test_dt_namespace():
    t = symbol('t', 'var * {when: datetime}')
    assert bdt.Ceil(t.when, 's').isidentical(t.when.dt.ceil('s'))
    assert bdt.date(t.when).isidentical(t.when.dt.date())
    assert bdt.day(t.when).isidentical(t.when.dt.day())
    assert bdt.dayofweek(t.when).isidentical(t.when.dt.dayofweek())
    assert bdt.dayofyear(t.when).isidentical(t.when.dt.dayofyear())
    assert bdt.days_in_month(t.when).isidentical(t.when.dt.days_in_month())
    assert bdt.Floor(t.when, 's').isidentical(t.when.dt.floor('s'))
    assert bdt.hour(t.when).isidentical(t.when.dt.hour())
    assert bdt.is_month_end(t.when).isidentical(t.when.dt.is_month_end())
    assert bdt.is_month_start(t.when).isidentical(t.when.dt.is_month_start())
    assert bdt.is_quarter_end(t.when).isidentical(t.when.dt.is_quarter_end())
    assert bdt.is_quarter_start(t.when).isidentical(t.when.dt.is_quarter_start())
    assert bdt.is_year_end(t.when).isidentical(t.when.dt.is_year_end())
    assert bdt.is_year_start(t.when).isidentical(t.when.dt.is_year_start())
    assert bdt.microsecond(t.when).isidentical(t.when.dt.microsecond())
    assert bdt.millisecond(t.when).isidentical(t.when.dt.millisecond())
    assert bdt.minute(t.when).isidentical(t.when.dt.minute())
    assert bdt.month(t.when).isidentical(t.when.dt.month())
    assert bdt.nanosecond(t.when).isidentical(t.when.dt.nanosecond())
    assert bdt.Round(t.when, 's').isidentical(t.when.dt.round('s'))
    assert bdt.quarter(t.when).isidentical(t.when.dt.quarter())
    assert bdt.second(t.when).isidentical(t.when.dt.second())
    assert bdt.strftime(t.when, 'format').isidentical(t.when.dt.strftime('format'))
    assert bdt.time(t.when).isidentical(t.when.dt.time())
    assert bdt.truncate(t.when, days=2).isidentical(t.when.dt.truncate(days=2))
    assert bdt.week(t.when).isidentical(t.when.dt.week())
    assert bdt.weekday(t.when).isidentical(t.when.dt.weekday())
    assert bdt.weekofyear(t.when).isidentical(t.when.dt.weekofyear())
    assert bdt.year(t.when).isidentical(t.when.dt.year())

def test_td_namespace():
    t = symbol('t', 'var * {span: timedelta}')
    assert bdt.nanoseconds(t.span).isidentical(t.span.dt.nanoseconds())
    assert bdt.seconds(t.span).isidentical(t.span.dt.seconds())
    assert bdt.total_seconds(t.span).isidentical(t.span.dt.total_seconds())

