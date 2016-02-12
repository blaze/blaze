from blaze.expr import symbol
from blaze.expr.datetime import isdatelike
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
    assert not isdatelike('int32')
    assert isdatelike('?date')
    assert not isdatelike('{is_outdated: bool}')


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
