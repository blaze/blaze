from __future__ import absolute_import, division, print_function

from blaze.compute.sql import compute, _compute
from blaze.objects.table import *
import sqlalchemy
import sqlalchemy as sa

t = Table('{name: string, amount: int, id: int}')

metadata = sa.MetaData()

s = sa.Table('accounts', metadata,
             sa.Column('name', sa.String),
             sa.Column('amount', sa.Integer),
             sa.Column('id', sa.Integer, primary_key=True),
             )

def normalize(s):
    return ' '.join(s.strip().split())

def test_table():
    result = str(compute(t, s))
    expected = """
    SELECT accounts.name, accounts.amount, accounts.id
    FROM accounts
    """.strip()

    assert normalize(result) == normalize(expected)



def test_projection():
    assert str(compute(t[['name', 'amount']], s)) == \
            str(sa.select([s.c.name, s.c.amount]))


def test_eq():
    assert str(_compute(t['amount'] == 100, s)) == str(s.c.amount == 100)


def test_selection():
    assert str(compute(t[t['amount'] == 0], s)) == \
            str(sa.select([s]).where(s.c.amount == 0))
    assert str(compute(t[t['amount'] > 150], s)) == \
            str(sa.select([s]).where(s.c.amount > 150))


def test_arithmetic():
    assert str(compute(t['amount'] + t['id'], s)) == \
            str(sa.select([s.c.amount + s.c.id]))
    assert str(_compute(t['amount'] + t['id'], s)) == str(s.c.amount + s.c.id)
    assert str(_compute(t['amount'] * t['id'], s)) == str(s.c.amount * s.c.id)

    assert str(compute(t['amount'] + t['id'] * 2, s)) == \
            str(sa.select([s.c.amount + s.c.id * 2]))
