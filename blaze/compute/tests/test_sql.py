from __future__ import absolute_import, division, print_function

from blaze.compute.sql import compute, computefull, select
from blaze.expr.table import *
import sqlalchemy
import sqlalchemy as sa
from blaze.compatibility import skip
from blaze.utils import unique

t = TableSymbol('{name: string, amount: int, id: int}')

metadata = sa.MetaData()

s = sa.Table('accounts', metadata,
             sa.Column('name', sa.String),
             sa.Column('amount', sa.Integer),
             sa.Column('id', sa.Integer, primary_key=True),
             )

tbig = TableSymbol('{name: string, sex: string[1], amount: int, id: int}')

sbig = sa.Table('accountsbig', metadata,
             sa.Column('name', sa.String),
             sa.Column('sex', sa.String),
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


def test_booleans():
    assert str(compute(t['name'] == 'Alice', s)) == str(s.c.name == 'Alice')

    assert str(compute((t['name'] == 'Alice') | (t['name'] == 'Bob'), s)) == \
                str((s.c.name == 'Alice') | (s.c.name == 'Bob'))


def test_join():
    lhs = sa.Table('amounts', metadata,
                   sa.Column('name', sa.String),
                   sa.Column('amount', sa.Integer))

    rhs = sa.Table('ids', metadata,
                   sa.Column('name', sa.String),
                   sa.Column('id', sa.Integer))

    expected = lhs.join(rhs, lhs.c.name == rhs.c.name)
    expected = select(list(unique(expected.columns, key=lambda c:
        c.name))).select_from(expected)

    L = TableSymbol('{name: string, amount: int}')
    R = TableSymbol('{name: string, id: int}')
    joined = Join(L, R, 'name')

    result = compute(joined, {L: lhs, R: rhs})

    assert str(result) == str(expected)

    assert str(select(result)) == str(select(expected))

    # Schemas match
    assert list(result.c.keys()) == list(joined.columns)


def test_unary_op():
    assert str(compute(exp(t['amount']), s)) == str(sa.func.exp(s.c.amount))


def test_reductions():
    assert str(compute(sum(t['amount']), s)) == \
            str(sa.sql.functions.sum(s.c.amount))
    assert str(compute(mean(t['amount']), s)) == \
            str(sa.sql.func.avg(s.c.amount))
    assert str(compute(count(t['amount']), s)) == \
            str(sa.sql.func.count(s.c.amount))

    assert 'amount' == compute(sum(t['amount']), s).name

def test_nunique():
    result = str(compute(nunique(t['amount']), s))

    assert 'distinct' in result.lower()
    assert 'count' in result.lower()
    assert 'amount' in result.lower()

    print(result)
    assert result == str(sa.sql.func.count(sa.distinct(s.c.amount)))

@skip("Fails because SQLAlchemy doesn't seem to know binary reductions")
def test_binary_reductions():
    assert str(compute(any(t['amount'] > 150), s)) == \
            str(sa.sql.functions.any(s.c.amount > 150))


def test_by():
    expr = By(t, t['name'], t['amount'].sum())
    result = compute(expr, s)
    expected = sa.select([s.c.name,
                          sa.sql.functions.sum(s.c.amount).label('amount')]
                         ).group_by(s.c.name)

    assert str(result) == str(expected)


def test_by_two():
    expr = By(tbig, tbig[['name', 'sex']], tbig['amount'].sum())
    result = compute(expr, sbig)
    expected = (sa.select([sbig.c.name,
                           sbig.c.sex,
                           sa.sql.functions.sum(sbig.c.amount).label('amount')])
                        .group_by(sbig.c.name, sbig.c.sex))

    assert str(result) == str(expected)


def test_by_three():
    result = compute(By(tbig,
                        tbig[['name', 'sex']],
                        (tbig['id'] + tbig['amount']).sum()),
                     sbig)

    expected = (sa.select([sbig.c.name,
                           sbig.c.sex,
                           sa.sql.functions.sum(sbig.c.id+ sbig.c.amount)])
                    .group_by(sbig.c.name, sbig.c.sex))

    assert str(result) == str(expected)


def test_join_projection():
    metadata = sa.MetaData()
    lhs = sa.Table('amounts', metadata,
                   sa.Column('name', sa.String),
                   sa.Column('amount', sa.Integer))

    rhs = sa.Table('ids', metadata,
                   sa.Column('name', sa.String),
                   sa.Column('id', sa.Integer))

    L = TableSymbol('{name: string, amount: int}')
    R = TableSymbol('{name: string, id: int}')
    want = Join(L, R, 'name')[['amount', 'id']]

    result = compute(want, {L: lhs, R: rhs})
    print(result)
    assert 'JOIN' in str(result)
    assert result.c.keys() == ['amount', 'id']
    assert 'amounts.name = ids.name' in str(result)


def test_sort():
    assert str(compute(t.sort('amount'), s)) == \
            str(select(s).order_by(s.c.amount))

    assert str(compute(t.sort('amount', ascending=False), s)) == \
            str(select(s).order_by(sqlalchemy.desc(s.c.amount)))


def test_head():
    assert str(compute(t.head(2), s)) == str(select(s).limit(2))
