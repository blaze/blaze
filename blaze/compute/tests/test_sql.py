from __future__ import absolute_import, division, print_function

from blaze.compute.sql import compute, computefull
from blaze.expr.table import *
import sqlalchemy
import sqlalchemy as sa
from blaze.py2help import skip

t = TableSymbol('{name: string, amount: int, id: int}')

metadata = sa.MetaData()

s = sa.Table('accounts', metadata,
             sa.Column('name', sa.String),
             sa.Column('amount', sa.Integer),
             sa.Column('id', sa.Integer, primary_key=True),
             )

def normalize(s):
    return ' '.join(s.strip().split())

def test_table():
    result = str(computefull(t, s))
    expected = """
    SELECT accounts.name, accounts.amount, accounts.id
    FROM accounts
    """.strip()

    assert normalize(result) == normalize(expected)



def test_projection():
    assert str(compute(t[['name', 'amount']], s)) == \
            str(sa.select([s.c.name, s.c.amount]))


def test_eq():
    assert str(compute(t['amount'] == 100, s)) == str(s.c.amount == 100)


def test_selection():
    assert str(compute(t[t['amount'] == 0], s)) == \
            str(sa.select([s]).where(s.c.amount == 0))
    assert str(compute(t[t['amount'] > 150], s)) == \
            str(sa.select([s]).where(s.c.amount > 150))


def test_arithmetic():
    assert str(computefull(t['amount'] + t['id'], s)) == \
            str(sa.select([s.c.amount + s.c.id]))
    assert str(compute(t['amount'] + t['id'], s)) == str(s.c.amount + s.c.id)
    assert str(compute(t['amount'] * t['id'], s)) == str(s.c.amount * s.c.id)

    assert str(computefull(t['amount'] + t['id'] * 2, s)) == \
            str(sa.select([s.c.amount + s.c.id * 2]))

def test_join():
    lhs = sa.Table('amounts', metadata,
                   sa.Column('name', sa.String),
                   sa.Column('amount', sa.Integer))

    rhs = sa.Table('ids', metadata,
                   sa.Column('name', sa.String),
                   sa.Column('id', sa.Integer))

    expected = lhs.join(rhs, lhs.c.name == rhs.c.name)

    L = TableSymbol('{name: string, amount: int}')
    R = TableSymbol('{name: string, id: int}')
    joined = Join(L, R, 'name')

    result = compute(joined, {L: lhs, R: rhs})

    assert str(result) == str(expected)

    assert str(sa.select([result])) == str(sa.select([expected]))


def test_unary_op():
    assert str(compute(exp(t['amount']), s)) == str(sa.func.exp(s.c.amount))


def test_reductions():
    assert str(compute(sum(t['amount']), s)) == \
            str(sa.sql.functions.sum(s.c.amount))
    assert str(compute(mean(t['amount']), s)) == \
            str(sa.sql.func.avg(s.c.amount))

@skip("Fails because SQLAlchemy doesn't seem to know binary reductions")
def test_binary_reductions():
    assert str(compute(any(t['amount'] > 150), s)) == \
            str(sa.sql.functions.any(s.c.amount > 150))
