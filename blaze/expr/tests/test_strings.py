import datashape
import pytest
from datashape import dshape

from blaze import symbol


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
