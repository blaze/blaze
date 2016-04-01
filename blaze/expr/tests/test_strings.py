import datashape
import pytest
from datashape import dshape

from blaze import symbol, broadcast_collect
from blaze.expr.strings import str_upper, str_lower, str_len


@pytest.mark.parametrize(
    'ds',
    [
        'var * {name: string}',
        'var * {name: ?string}',
        'var * string',
        'var * ?string',
        'string',
    ]
)
def test_like(ds):
    t = symbol('t', ds)
    expr = getattr(t, 'name', t).like('Alice*')
    assert expr.pattern == 'Alice*'
    assert expr.schema.measure == dshape(
        '%sbool' % ('?' if '?' in ds else '')
    ).measure


def test_broadcasting_string_funcs():
    t = symbol('t', 'var * {name: string, val: int32}')
    t_ = symbol('t', t.dshape.measure)
    bc = broadcast_collect(t.name.str_upper().str_len() +
                           t.name.str_lower().str_len() *
                           t.val)
    assert bc._children[0].isidentical(t)
    assert bc._scalars[0].isidentical(t_)
    assert bc._scalar_expr == (str_len(_child=str_upper(_child=t_.name)) +
                               str_len(_child=str_lower(_child=t_.name)) *
                               t_.val)
