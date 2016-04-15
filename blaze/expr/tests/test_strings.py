import datashape
import pytest
from datashape import dshape

from blaze import symbol, broadcast_collect
from blaze.expr.strings import str_upper, str_lower, str_len

dshapes = ['var * {name: string}',
           'var * {name: ?string}',
           'var * string',
           'var * ?string',
           'string']

@pytest.mark.parametrize('ds', dshapes)
def test_like(ds):
    t = symbol('t', ds)
    expr = getattr(t, 'name', t).like('Alice*')
    assert expr.pattern == 'Alice*'
    expected = dshape('{}bool'.format('?' if '?' in ds else ''))
    assert expr.schema.measure == expected.measure


@pytest.mark.parametrize('ds', dshapes)
def test_str_upper_schema(ds):
    t = symbol('t', ds)
    expr_upper = getattr(t, 'name', t).str_upper()
    expr_lower = getattr(t, 'name', t).str_upper()
    assert (expr_upper.schema.measure ==
            expr_lower.schema.measure ==
            dshape('{}string'.format('?' if '?' in ds else '')).measure)


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
