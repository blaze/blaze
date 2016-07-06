from datashape import dshape
from datashape.util.testing import assert_dshape_equal
import pytest

from blaze import symbol
import blaze as bz
import blaze.expr.strings as bzs

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
    blaze symbol used to test exceptions raised by cat()
    '''
    ds = dshape('3 * {name: string, comment: string, num: int32}')
    s = symbol('s', dshape=ds)
    return s


@pytest.mark.parametrize('ds', dshapes)
def test_like(ds):
    t = symbol('t', ds)
    expr = getattr(t, 'name', t).str.like('Alice*')
    assert expr.pattern == 'Alice*'
    assert_dshape_equal(
        expr.schema.measure,
        dshape('%sbool' % ('?' if '?' in ds else '')).measure,
    )


@pytest.mark.parametrize('ds', dshapes)
def test_upper_schema(ds):
    t = symbol('t', ds)
    expr_upper = getattr(t, 'name', t).str.upper()
    expr_lower = getattr(t, 'name', t).str.lower()
    assert (expr_upper.schema.measure ==
            expr_lower.schema.measure ==
            dshape('%sstring' % ('?' if '?' in ds else '')).measure)


def test_str_namespace():
    t = symbol('t', 'var * {name: string}')
    assert bzs.upper(t.name).isidentical(t.name.str.upper())
    assert bzs.lower(t.name).isidentical(t.name.str.lower())
    assert (bzs.lower(bzs.upper(t.name))
            .isidentical(t.name.str.upper().str.lower()))
    assert bzs.len(t.name).isidentical(t.name.str.len())
    assert bzs.like(t.name, '*a').isidentical(t.name.str.like('*a'))
    assert (bzs.cat(bzs.cat(t.name, t.name, sep=' ++ '), t.name)
            .isidentical(t.name.str.cat(t.name, sep=' ++ ')
                               .str.cat(t.name)))
    assert bzs.isalnum(t.name).isidentical(t.name.str.isalnum())
    assert bzs.isalpha(t.name).isidentical(t.name.str.isalpha())
    assert bzs.isdecimal(t.name).isidentical(t.name.str.isdecimal())
    assert bzs.isdigit(t.name).isidentical(t.name.str.isdigit())
    assert bzs.islower(t.name).isidentical(t.name.str.islower())
    assert bzs.isnumeric(t.name).isidentical(t.name.str.isnumeric())
    assert bzs.isspace(t.name).isidentical(t.name.str.isspace())
    assert bzs.istitle(t.name).isidentical(t.name.str.istitle())
    assert bzs.isupper(t.name).isidentical(t.name.str.isupper())

    assert bzs.replace(t.name, 'A', 'a').isidentical(t.name.str.replace('A', 'a'))
    assert bzs.capitalize(t.name).isidentical(t.name.str.capitalize())
    assert bzs.strip(t.name).isidentical(t.name.str.strip())
    assert bzs.lstrip(t.name).isidentical(t.name.str.lstrip())
    assert bzs.rstrip(t.name).isidentical(t.name.str.rstrip())
    assert bzs.pad(t.name, 5).isidentical(t.name.str.pad(5))
    assert (bzs.slice_replace(t.name, 1, 3, 'foo')
            .isidentical(t.name.str.slice_replace(1, 3, 'foo')))

@pytest.mark.parametrize('ds', lhsrhs_ds)
def test_cat_schema_shape(ds):
    t = symbol('t', ds)
    expr = t.name.str.cat(t.comment)
    assert (expr.schema.measure ==
            dshape('%sstring' % ('?' if '?' in ds else '')).measure)
    assert expr.lhs.shape == expr.rhs.shape == expr.shape


def test_cat_exception_non_string_sep(strcat_sym):
    with pytest.raises(TypeError):
        strcat_sym.name.str.cat(strcat_sym.comment, sep=123)


def test_cat_exception_non_string_col_to_cat(strcat_sym):
    with pytest.raises(TypeError):
        strcat_sym.name.str.cat(strcat_sym.num)

