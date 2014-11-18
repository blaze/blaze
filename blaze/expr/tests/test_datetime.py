from blaze.expr import Symbol
from blaze.expr.datetime import isdatelike
from blaze.compatibility import builtins
from datashape import dshape

import pytest

def test_datetime_dshape():
    t = Symbol('t', '5 * {name: string, when: datetime}')
    assert t.when.day.dshape == dshape('5 * int32')
    assert t.when.date.dshape == dshape('5 * date')


def test_date_attribute():
    t = Symbol('t', '5 * {name: string, when: datetime}')
    expr = t.when.day
    assert eval(str(expr)).isidentical(expr)


def test_invalid_date_attribute():
    t = Symbol('t', '5 * {name: string, when: datetime}')
    with pytest.raises(AttributeError):
        t.name.day


def test_date_attribute_completion():
    t = Symbol('t', '5 * {name: string, when: datetime}')
    assert 'day' in dir(t.when)
    assert 'day' not in dir(t.name)
    assert not builtins.all([x.startswith('__') and x.endswith('__')
                            for x in dir(t.name)])

def test_datetime_attribute_name():
    t = Symbol('t', '5 * {name: string, when: datetime}')
    assert 'when' in t.when.day._name


def test_utcfromtimestamp():
    t = Symbol('t', '10 * {name: string, ts: int32}')
    assert t.ts.utcfromtimestamp.dshape == dshape('10 * datetime')


def test_isdatelike():
    assert not isdatelike('int32')
    assert isdatelike('?date')
    assert not isdatelike('{is_outdated: bool}')
