from __future__ import absolute_import, division, print_function

from blaze.compute.sql import compute, computefull, select
from blaze import SQL
from blaze.expr import *
import sqlalchemy
import sqlalchemy as sa
from blaze.compatibility import xfail
from blaze.utils import unique

t = TableSymbol('t', '{name: string, amount: int, id: int}')

metadata = sa.MetaData()

s = sa.Table('accounts', metadata,
             sa.Column('name', sa.String),
             sa.Column('amount', sa.Integer),
             sa.Column('id', sa.Integer, primary_key=True),
             )

tbig = TableSymbol('tbig', '{name: string, sex: string[1], amount: int, id: int}')

sbig = sa.Table('accountsbig', metadata,
             sa.Column('name', sa.String),
             sa.Column('sex', sa.String),
             sa.Column('amount', sa.Integer),
             sa.Column('id', sa.Integer, primary_key=True),
             )

def normalize(s):
    return ' '.join(s.strip().split()).lower()

def test_table():
    result = str(computefull(t, s))
    expected = """
    SELECT accounts.name, accounts.amount, accounts.id
    FROM accounts
    """.strip()

    assert normalize(result) == normalize(expected)



def test_projection():
    print(compute(t[['name', 'amount']], s))
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
    metadata = sa.MetaData()
    lhs = sa.Table('amounts', metadata,
                   sa.Column('name', sa.String),
                   sa.Column('amount', sa.Integer))

    rhs = sa.Table('ids', metadata,
                   sa.Column('name', sa.String),
                   sa.Column('id', sa.Integer))

    expected = lhs.join(rhs, lhs.c.name == rhs.c.name)
    expected = select(list(unique(expected.columns, key=lambda c:
        c.name))).select_from(expected)

    L = TableSymbol('L', '{name: string, amount: int}')
    R = TableSymbol('R', '{name: string, id: int}')
    joined = join(L, R, 'name')

    result = compute(joined, {L: lhs, R: rhs})

    assert str(result) == str(expected)

    assert str(select(result)) == str(select(expected))

    # Schemas match
    assert list(result.c.keys()) == list(joined.columns)


def test_multi_column_join():
    metadata = sa.MetaData()
    lhs = sa.Table('aaa', metadata,
                   sa.Column('x', sa.Integer),
                   sa.Column('y', sa.Integer),
                   sa.Column('z', sa.Integer))

    rhs = sa.Table('bbb', metadata,
                   sa.Column('w', sa.Integer),
                   sa.Column('x', sa.Integer),
                   sa.Column('y', sa.Integer))

    L = TableSymbol('L', '{x: int, y: int, z: int}')
    R = TableSymbol('R', '{w: int, x: int, y: int}')
    joined = join(L, R, ['x', 'y'])

    expected = lhs.join(rhs, (lhs.c.x == rhs.c.x)
                           & (lhs.c.y == rhs.c.y))
    expected = select(list(unique(expected.columns, key=lambda c:
        c.name))).select_from(expected)

    result = compute(joined, {L: lhs, R: rhs})

    assert str(result) == str(expected)

    assert str(select(result)) == str(select(expected))

    # Schemas match
    print(result.c.keys())
    print(joined.columns)
    assert list(result.c.keys()) == list(joined.columns)


def test_unary_op():
    assert str(compute(exp(t['amount']), s)) == str(sa.func.exp(s.c.amount))


def test_unary_op():
    assert str(compute(-t['amount'], s)) == str(-s.c.amount)


def test_reductions():
    assert str(compute(sum(t['amount']), s)) == \
            str(sa.sql.functions.sum(s.c.amount))
    assert str(compute(mean(t['amount']), s)) == \
            str(sa.sql.func.avg(s.c.amount))
    assert str(compute(count(t['amount']), s)) == \
            str(sa.sql.func.count(s.c.amount))

    assert 'amount_sum' == compute(sum(t['amount']), s).name

def test_count_on_table():
    assert normalize(str(select(compute(t.count(), s)))) == normalize("""
    SELECT count(accounts.id) as tbl_row_count
    FROM accounts""")

    assert normalize(str(select(compute(t[t.amount > 0].count(), s)))) == \
    normalize("""
    SELECT count(accounts.id) as tbl_row_count
    FROM accounts
    WHERE accounts.amount > """)

def test_distinct():
    result = str(compute(Distinct(t['amount']), s))

    assert 'distinct' in result.lower()
    assert 'amount' in result.lower()

    print(result)
    assert result == str(sa.distinct(s.c.amount))


def test_nunique():
    result = str(computefull(nunique(t['amount']), s))

    print(result)
    assert 'distinct' in result.lower()
    assert 'count' in result.lower()
    assert 'amount' in result.lower()


@xfail(reason="Fails because SQLAlchemy doesn't seem to know binary reductions")
def test_binary_reductions():
    assert str(compute(any(t['amount'] > 150), s)) == \
            str(sqlalchemy.sql.functions.any(s.c.amount > 150))


def test_by():
    expr = by(t['name'], t['amount'].sum())
    result = compute(expr, s)
    expected = sa.select([s.c.name,
                          sa.sql.functions.sum(s.c.amount).label('amount_sum')]
                         ).group_by(s.c.name)

    assert str(result) == str(expected)


def test_by_head():
    t2 = t.head(100)
    expr = by(t2['name'], t2['amount'].sum())
    result = compute(expr, s)
    s2 = select(s).limit(100)
    # expected = sa.select([s2.c.name,
    #                       sa.sql.functions.sum(s2.c.amount).label('amount_sum')]
    #                      ).group_by(s2.c.name)
    expected = """
    SELECT accounts.name, sum(accounts.amount) as amount_sum
    FROM accounts
    GROUP by accounts.name
    LIMIT :param_1"""
    assert normalize(str(result)) == normalize(str(expected))


def test_by_two():
    expr = by(tbig[['name', 'sex']], tbig['amount'].sum())
    result = compute(expr, sbig)
    expected = (sa.select([sbig.c.name,
                           sbig.c.sex,
                           sa.sql.functions.sum(sbig.c.amount).label('amount_sum')])
                        .group_by(sbig.c.name, sbig.c.sex))

    assert str(result) == str(expected)


def test_by_three():
    result = compute(by(tbig[['name', 'sex']],
                        (tbig['id'] + tbig['amount']).sum()),
                     sbig)

    expected = (sa.select([sbig.c.name,
                           sbig.c.sex,
                           sa.sql.functions.sum(sbig.c.id+ sbig.c.amount)])
                    .group_by(sbig.c.name, sbig.c.sex))

    assert str(result) == str(expected)

def test_by_summary_clean():
    expr = by(t.name, min=t.amount.min(), max=t.amount.max())
    result = compute(expr, s)

    expected = """
    SELECT accounts.name, max(accounts.amount) AS max, min(accounts.amount) AS min
    FROM accounts
    GROUP BY accounts.name
    """

    assert normalize(str(result)) == normalize(expected)




def test_join_projection():
    metadata = sa.MetaData()
    lhs = sa.Table('amounts', metadata,
                   sa.Column('name', sa.String),
                   sa.Column('amount', sa.Integer))

    rhs = sa.Table('ids', metadata,
                   sa.Column('name', sa.String),
                   sa.Column('id', sa.Integer))

    L = TableSymbol('L', '{name: string, amount: int}')
    R = TableSymbol('R', '{name: string, id: int}')
    want = join(L, R, 'name')[['amount', 'id']]

    result = compute(want, {L: lhs, R: rhs})
    print(result)
    assert 'join' in str(result).lower()
    assert result.c.keys() == ['amount', 'id']
    assert 'amounts.name = ids.name' in str(result)


def test_sort():
    assert str(compute(t.sort('amount'), s)) == \
            str(select(s).order_by(s.c.amount))

    assert str(compute(t.sort('amount', ascending=False), s)) == \
            str(select(s).order_by(sqlalchemy.desc(s.c.amount)))


def test_head():
    assert str(compute(t.head(2), s)) == str(select(s).limit(2))


def test_label():
    assert str(compute((t['amount'] * 10).label('foo'), s)) == \
            str((s.c.amount * 10).label('foo'))


def test_relabel():
    result = compute(t.relabel({'name': 'NAME', 'id': 'ID'}), s)
    expected = select([s.c.name.label('NAME'), s.c.amount, s.c.id.label('ID')])

    assert str(result) == str(expected)


def test_merge():
    col = (t['amount'] * 2).label('new')

    expr = merge(t['name'], col)

    result = str(compute(expr, s))

    assert 'amount * ' in result
    assert 'FROM accounts' in result
    assert 'SELECT accounts.name' in result
    assert 'new' in result

def test_projection_of_selection():
    print(compute(t[t['amount'] < 0][['name', 'amount']], s))
    assert len(str(compute(t[t['amount'] < 0], s))) > \
            len(str(compute(t[t['amount'] < 0][['name', 'amount']], s)))


def test_union():
    ts = [TableSymbol('t_%d' % i, '{name: string, amount: int, id: int}')
            for i in [1, 2, 3]]
    ss = [sa.Table('accounts_%d' % i, metadata,
             sa.Column('name', sa.String),
             sa.Column('amount', sa.Integer),
             sa.Column('id', sa.Integer, primary_key=True)) for i in [1, 2, 3]]

    expr = union(*ts)

    result = str(select(compute(expr, dict(zip(ts, ss)))))

    assert "SELECT name, amount, id" in str(result)
    assert "accounts_1 UNION accounts_2 UNION accounts_3" in str(result)


def test_outer_join():
    L = TableSymbol('L', '{id: int, name: string, amount: real}')
    R = TableSymbol('R', '{city: string, id: int}')

    from blaze.sql import SQL
    engine = sa.create_engine('sqlite:///:memory:')

    _left = [(1, 'Alice', 100),
            (2, 'Bob', 200),
            (4, 'Dennis', 400)]
    left = SQL(engine, 'left', schema=L.schema)
    left.extend(_left)

    _right = [('NYC', 1),
             ('Boston', 1),
             ('LA', 3),
             ('Moscow', 4)]
    right = SQL(engine, 'right', schema=R.schema)
    right.extend(_right)

    conn = engine.connect()


    query = compute(join(L, R, how='inner'), {L: left.table, R: right.table})
    result = list(map(tuple, conn.execute(query).fetchall()))

    assert set(result) == set(
            [(1, 'Alice', 100, 'NYC'),
             (1, 'Alice', 100, 'Boston'),
             (4, 'Dennis', 400, 'Moscow')])

    query = compute(join(L, R, how='left'), {L: left.table, R: right.table})
    result = list(map(tuple, conn.execute(query).fetchall()))

    assert set(result) == set(
            [(1, 'Alice', 100, 'NYC'),
             (1, 'Alice', 100, 'Boston'),
             (2, 'Bob', 200, None),
             (4, 'Dennis', 400, 'Moscow')])

    query = compute(join(L, R, how='right'), {L: left.table, R: right.table})
    print(query)
    result = list(map(tuple, conn.execute(query).fetchall()))
    print(result)

    assert set(result) == set(
            [(1, 'Alice', 100, 'NYC'),
             (1, 'Alice', 100, 'Boston'),
             (3, None, None, 'LA'),
             (4, 'Dennis', 400, 'Moscow')])

    # SQLAlchemy doesn't support full outer join
    """
    query = compute(join(L, R, how='outer'), {L: left.table, R: right.table})
    result = list(map(tuple, conn.execute(query).fetchall()))

    assert set(result) == set(
            [(1, 'Alice', 100, 'NYC'),
             (1, 'Alice', 100, 'Boston'),
             (2, 'Bob', 200, None),
             (3, None, None, 'LA'),
             (4, 'Dennis', 400, 'Moscow')])
    """

    conn.close()


def test_summary():
    expr = summary(a=t.amount.sum(), b=t.id.count())
    result = str(compute(expr, s))

    assert 'sum(accounts.amount) as a' in result.lower()
    assert 'count(accounts.id) as b' in result.lower()


def test_summary_by():
    expr = by(t.name, summary(a=t.amount.sum(), b=t.id.count()))

    result = str(compute(expr, s))

    assert 'sum(accounts.amount) as a' in result.lower()
    assert 'count(accounts.id) as b' in result.lower()

    assert 'group by accounts.name' in result.lower()


def test_clean_join():
    name = sa.Table('name', metadata,
             sa.Column('id', sa.Integer),
             sa.Column('name', sa.String),
             )
    city = sa.Table('place', metadata,
             sa.Column('id', sa.Integer),
             sa.Column('city', sa.String),
             sa.Column('country', sa.String),
             )
    friends = sa.Table('friends', metadata,
             sa.Column('a', sa.Integer),
             sa.Column('b', sa.Integer),
             )

    tcity = TableSymbol('city', discover(city))
    tfriends = TableSymbol('friends', discover(friends))
    tname = TableSymbol('name', discover(name))

    ns = {tname: name, tfriends: friends, tcity: city}

    expr = join(tfriends, tname, 'a', 'id')
    assert normalize(str(compute(expr, ns))) == normalize("""
    SELECT friends.a, friends.b, name.name
    FROM friends JOIN name on friends.a = name.id""")


    expr = join(join(tfriends, tname, 'a', 'id'), tcity, 'a', 'id')
    assert normalize(str(compute(expr, ns))) == normalize("""
    SELECT a, b, name, place.city, place.country
    FROM (SELECT friends.a as a, friends.b as b, name.name as name
          FROM friends JOIN name ON friends.a = name.id)
    JOIN place on friends.a = place.id""")


def test_like():
    expr = t.like(name='Alice*')
    assert normalize(str(compute(expr, s))) == normalize("""
    SELECT accounts.name, accounts.amount, accounts.id
    FROM accounts
    WHERE accounts.name LIKE :name_1""")

def test_reductions_on_complex_selections():
    assert normalize(str(select(compute(t[t.amount > 0].id.sum(), s)))) == \
            normalize("""
    SELECT sum(accounts.id)
    FROM accounts
    WHERE accounts.amount > 0 """)
