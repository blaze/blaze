import datashape
import pytest
from datashape import dshape

from blaze import symbol

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
    assert expr.schema.measure == dshape(
        '%sbool' % ('?' if '?' in ds else '')
    ).measure


@pytest.mark.parametrize('ds', dshapes)
def test_str_upper_schema(ds):
    t = symbol('t', ds)
    expr_upper = getattr(t, 'name', t).str_upper()
    expr_lower = getattr(t, 'name', t).str_upper()
    assert (expr_upper.schema.measure ==
            expr_lower.schema.measure ==
            dshape('%sstring' % ('?' if '?' in ds else '')).measure)
