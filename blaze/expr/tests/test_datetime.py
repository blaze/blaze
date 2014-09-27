from blaze.expr import TableSymbol
from blaze.compatibility import builtins
from datashape import dshape

import pytest

def test_datetime_dshape():
    t = TableSymbol('t', '5 * {name: string, when: datetime}')
    assert t.when.day.dshape == dshape('5 * int32')
    assert t.when.date.dshape == dshape('5 * date')


def test_date_attribute():
    t = TableSymbol('t', '{name: string, when: datetime}')
    expr = t.when.day
    assert eval(str(expr)).isidentical(expr)


def test_invalid_date_attribute():
    t = TableSymbol('t', '{name: string, when: datetime}')
    with pytest.raises(AttributeError):
        t.name.day


def test_date_attribute_completion():
    t = TableSymbol('t', '{name: string, when: datetime}')
    assert 'day' in dir(t.when)
    assert 'day' not in dir(t.name)
    assert not builtins.all([x.startswith('__') and x.endswith('__')
                            for x in dir(t.name)])

