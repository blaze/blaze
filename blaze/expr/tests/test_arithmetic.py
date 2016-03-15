from itertools import product
import pytest

from blaze.expr import Add, USub, Not, Gt, Mult, Concat, Interp, Repeat, \
    Relational
from blaze import symbol
from datashape import dshape, Option


x = symbol('x', '5 * 3 * int32')
y = symbol('y', '5 * 3 * int32')
w = symbol('w', '5 * 3 * float32')
z = symbol('z', '5 * 3 * int64')
a = symbol('a', 'int32')
b = symbol('b', '5 * 3 * bool')
cs = symbol('cs', 'string')
d = symbol('d', '5*3*?int32')
e = symbol('e', '?int32')
f = symbol('f', '?bool')
g = symbol('g', 'datetime')
h = symbol('h', 'datetime')
i = symbol('i', '?datetime')
j = symbol('j', '?datetime')
k = symbol('k', 'timedelta')
l = symbol('l', '?timedelta')
optionals = {d, e, f}


def test_arithmetic_dshape_on_collections():
    assert Add(x, y).shape == x.shape == y.shape


def test_arithmetic_broadcasts_to_scalars():
    assert Add(x, a).shape == x.shape
    assert Add(x, 1).shape == x.shape


def test_unary_ops_are_elemwise():
    assert USub(x).shape == x.shape
    assert Not(b).shape == b.shape


@pytest.mark.parametrize('sym', (b, f))
def test_not_dshape(sym):
    if sym in optionals:
        assert Not(sym).schema == dshape(Option('bool'))
    else:
        assert Not(sym).schema == dshape('bool')


@pytest.mark.parametrize('relation_type', Relational.__subclasses__())
def test_relations_maintain_shape(relation_type):
    assert relation_type(x, y).shape == x.shape


@pytest.mark.parametrize('relation_type', Relational.__subclasses__())
@pytest.mark.parametrize('lhs,rhs', product((x, y, d, e, f), repeat=2))
def test_relations_are_boolean(lhs, rhs, relation_type):
    if lhs in optionals or rhs in optionals:
        assert relation_type(lhs, rhs).schema == dshape(Option('bool'))
    else:
        assert relation_type(lhs, rhs).schema == dshape('bool')


def test_names():
    assert Add(x, 1)._name == x._name
    assert Add(1, x)._name == x._name
    assert Mult(Add(1, x), 2)._name == x._name

    assert Add(y, x)._name != x._name
    assert Add(y, x)._name != y._name

    assert Add(x, x)._name == x._name


def test_inputs():
    assert (x + y)._inputs == (x, y)
    assert (x + 1)._inputs == (x,)
    assert (1 + y)._inputs == (y,)


def test_printing():
    assert str(-x) == '-x'
    assert str(-(x + y)) == '-(x + y)'

    assert str(~b) == '~b'
    assert str(~(b | (x > y))) == '~(b | (x > y))'


def test_dir():
    i = symbol('i', '10 * int')
    d = symbol('d', '10 * datetime')

    assert isinstance(i + 1, Add)  # this works
    with pytest.raises(Exception):  # this doesn't
        d + 1


def test_arith_ops_promote_dtype():
    r = w + z
    assert r.dshape == dshape('5 * 3 * float64')


def test_str_arith():
    rep = cs * 1
    assert isinstance(rep, Repeat)
    assert rep.lhs.isidentical(cs)
    assert rep.rhs == 1

    with pytest.raises(TypeError):
        cs * 1.5

    with pytest.raises(TypeError):
        cs * 'a'

    interp = cs % cs
    assert isinstance(interp, Interp)
    assert interp.lhs.isidentical(cs)
    assert interp.rhs.isidentical(cs)

    interp_lit = cs % 'a'
    assert isinstance(interp_lit, Interp)
    assert interp_lit.lhs.isidentical(cs)
    assert interp_lit.rhs == 'a'

    with pytest.raises(Exception):
        cs / 1

    with pytest.raises(Exception):
        cs // 1


def test_datetime_sub():
    assert (g - h).dshape == dshape('timedelta')
    assert (g - i).dshape == dshape('?timedelta')
    assert (i - j).dshape == dshape('?timedelta')
    assert (g - k).dshape == dshape('datetime')
    assert (g - l).dshape == dshape('?datetime')
    assert (i - k).dshape == dshape('?datetime')
    assert (i - l).dshape == dshape('?datetime')


def test_datetime_add():
    assert (g + k).dshape == dshape('datetime')
    assert (g + l).dshape == dshape('?datetime')
    assert (i + k).dshape == dshape('?datetime')
    assert (i + l).dshape == dshape('?datetime')
