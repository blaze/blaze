import pytest
from datashape import dshape

from blaze import symbol

dshapes = ['var * {name: string}',
           'var * {name: ?string}',
           'var * string',
           'var * ?string',
           'string']

lhsrhs_ds = ['var * {name: string, comment: string[25]}',
             'var * {name: string[10], comment: string}',
             'var * {name: string, comment: string}',
             'var * {name: ?string, comment: string}',
             'var * {name: string, comment: ?string}',
             '10 * {name: string, comment: ?string}']


@pytest.fixture(scope='module')
def strcat_sym():
    '''
    blaze symbol used to test exceptions raised by str_cat()
    '''
    ds = dshape('3 * {name: string, comment: string, num: int32}')
    s = symbol('s', dshape=ds)
    return s


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


@pytest.mark.parametrize('ds', lhsrhs_ds)
def test_str_cat_schema_shape(ds):
    t = symbol('t', ds)
    expr = t.name.str_cat(t.comment)
    assert (expr.schema.measure ==
            dshape('%sstring' % ('?' if '?' in ds else '')).measure)
    assert expr.lhs.shape == expr.rhs.shape == expr.shape


def test_str_cat_exception_non_string_sep(strcat_sym):
    with pytest.raises(TypeError):
        strcat_sym.name.str_cat(strcat_sym.comment, sep=123)


def test_str_cat_exception_non_string_col_to_cat(strcat_sym):
    with pytest.raises(TypeError):
        strcat_sym.name.str_cat(strcat_sym.num)
