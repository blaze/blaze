from blaze.expr import TableSymbol
from blaze.compatibility import builtins

import pytest

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

