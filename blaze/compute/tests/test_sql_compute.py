from __future__ import absolute_import, division, print_function

import pytest

sa = pytest.importorskip('sqlalchemy')

import itertools
import sqlite3
from distutils.version import LooseVersion


import datashape
from odo import into, discover
import numpy as np
import pandas as pd
import pandas.util.testing as tm
from pandas import DataFrame
from toolz import unique

from odo import odo, drop, resource
from blaze import data as bz_data
from blaze.compatibility import xfail
from blaze.compute.sql import compute, select, lower_column, compute_up
from blaze.expr import (
    by,
    coalesce,
    cos,
    count,
    datetime as bz_datetime,
    exp,
    floor,
    greatest,
    join,
    least,
    mean,
    merge,
    nunique,
    sin,
    sum,
    summary,
    symbol,
    transform,
)
from blaze.utils import tmpfile, example, normalize, literal_compile


def computefull(t, s):
    return select(compute(t, s, return_type='native'))


names = ('tbl%d' % i for i in itertools.count())


@pytest.fixture(scope='module')
def city_data():
    # make the engine
    engine = sa.create_engine('sqlite:///:memory:')
    metadata = sa.MetaData(engine)

    # name table
    name = sa.Table('name', metadata,
                    sa.Column('id', sa.Integer),
                    sa.Column('name', sa.String),
                    )
    name.create()

    # city table
    city = sa.Table('city', metadata,
                    sa.Column('id', sa.Integer),
                    sa.Column('city', sa.String),
                    sa.Column('country', sa.String),
                    )
    city.create()

    s = symbol('s', discover(engine))
    return {'engine': engine, 'metadata': metadata, 'name': name, 'city': city,
            's': s}


t = symbol('t', 'var * {name: string, amount: int, id: int}')
t_str_cat = symbol('t',
                   """var * {name: string,
                             amount: int,
                             id: int,
                             comment: string,
                             product: string}""")

nt = symbol('t', 'var * {name: ?string, amount: float64, id: int}')

metadata = sa.MetaData()

s = sa.Table('accounts', metadata,
             sa.Column('name', sa.String),
             sa.Column('amount', sa.Integer),
             sa.Column('id', sa.Integer, primary_key=True))

s_str_cat = sa.Table('accounts2', metadata,
                     sa.Column('name', sa.String),
                     sa.Column('amount', sa.Integer),
                     sa.Column('id', sa.Integer, primary_key=True),
                     sa.Column('comment', sa.String),
                     sa.Column('product', sa.String))

tdate = symbol('t',
               """var * {
                    name: string,
                    amount: int,
                    id: int,
                    occurred_on: datetime
                }""")

ns = sa.Table('nullaccounts', metadata,
              sa.Column('name', sa.String, nullable=True),
              sa.Column('amount', sa.REAL),
              sa.Column('id', sa.Integer, primary_key=True),
              )

sdate = sa.Table('accdate', metadata,
                 sa.Column('name', sa.String),
                 sa.Column('amount', sa.Integer),
                 sa.Column('id', sa.Integer, primary_key=True),
                 sa.Column('occurred_on', sa.DateTime))


tbig = symbol('tbig',
              'var * {name: string, sex: string[1], amount: int, id: int}')

sbig = sa.Table('accountsbig', metadata,
                sa.Column('name', sa.String),
                sa.Column('sex', sa.String),
                sa.Column('amount', sa.Integer),
                sa.Column('id', sa.Integer, primary_key=True))


def test_table():
    result = str(computefull(t, s))
    expected = """
    SELECT accounts.name, accounts.amount, accounts.id
    FROM accounts
    """.strip()

    assert normalize(result) == normalize(expected)


def test_projection():
    print(compute(t[['name', 'amount']], s, return_type='native'))
    assert str(compute(t[['name', 'amount']], s, return_type='native')) == \
        str(sa.select([s.c.name, s.c.amount]))


def test_eq():
    assert str(compute(t['amount'] == 100, s, post_compute=False, return_type='native')) == \
        str(s.c.amount == 100)


def test_eq_unicode():
    assert str(compute(t['name'] == u'Alice', s, post_compute=False, return_type='native')) == \
        str(s.c.name == u'Alice')


def test_selection():
    assert str(compute(t[t['amount'] == 0], s, return_type='native')) == \
        str(sa.select([s]).where(s.c.amount == 0))
    assert str(compute(t[t['amount'] > 150], s, return_type='native')) == \
        str(sa.select([s]).where(s.c.amount > 150))


def test_arithmetic():
    assert str(compute(t['amount'] + t['id'], s, return_type='native')) == \
        str(sa.select([s.c.amount + s.c.id]))
    assert str(compute(t['amount'] + t['id'], s, post_compute=False, return_type='native')) == \
        str(s.c.amount + s.c.id)
    assert str(compute(t['amount'] * t['id'], s, post_compute=False, return_type='native')) == \
        str(s.c.amount * s.c.id)

    assert str(compute(t['amount'] * 2, s, post_compute=False, return_type='native')) == \
        str(s.c.amount * 2)
    assert str(compute(2 * t['amount'], s, post_compute=False, return_type='native')) == \
        str(2 * s.c.amount)

    assert (str(compute(~(t['amount'] > 10), s, post_compute=False, return_type='native')) ==
            "accounts.amount <= :amount_1")

    assert str(compute(t['amount'] + t['id'] * 2, s, return_type='native')) == \
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

    L = symbol('L', 'var * {name: string, amount: int}')
    R = symbol('R', 'var * {name: string, id: int}')
    joined = join(L, R, 'name')

    result = compute(joined, {L: lhs, R: rhs}, return_type='native')

    assert normalize(str(result)) == normalize("""
    SELECT amounts.name, amounts.amount, ids.id
    FROM amounts JOIN ids ON amounts.name = ids.name""")

    assert str(select(result)) == str(select(expected))

    # Schemas match
    assert list(result.c.keys()) == list(joined.fields)

    # test sort on join

    result = compute(joined.sort('amount'), {L: lhs, R: rhs}, return_type='native')
    assert normalize(str(result)) == normalize("""
     select
        anon_1.name,
        anon_1.amount,
        anon_1.id
    from (select
              amounts.name as name,
              amounts.amount as amount,
              ids.id as id
          from
              amounts
          join
              ids
          on
              amounts.name = ids.name) as anon_1
    order by
        anon_1.amount asc""")


def test_clean_complex_join():
    metadata = sa.MetaData()
    lhs = sa.Table('amounts', metadata,
                   sa.Column('name', sa.String),
                   sa.Column('amount', sa.Integer))

    rhs = sa.Table('ids', metadata,
                   sa.Column('name', sa.String),
                   sa.Column('id', sa.Integer))

    L = symbol('L', 'var * {name: string, amount: int}')
    R = symbol('R', 'var * {name: string, id: int}')

    joined = join(L[L.amount > 0], R, 'name')

    result = compute(joined, {L: lhs, R: rhs}, return_type='native')

    expected1 = """
        SELECT amounts.name, amounts.amount, ids.id
        FROM amounts JOIN ids ON amounts.name = ids.name
        WHERE amounts.amount > :amount_1"""

    expected2 = """
        SELECT alias.name, alias.amount, ids.id
        FROM (SELECT amounts.name AS name, amounts.amount AS amount
              FROM amounts
              WHERE amounts.amount > :amount_1) AS alias
        JOIN ids ON alias.name = ids.name"""

    assert (normalize(str(result)) == normalize(expected1) or
            normalize(str(result)) == normalize(expected2))


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

    L = symbol('L', 'var * {x: int, y: int, z: int}')
    R = symbol('R', 'var * {w: int, x: int, y: int}')
    joined = join(L, R, ['x', 'y'])

    expected = lhs.join(rhs, (lhs.c.x == rhs.c.x)
                        & (lhs.c.y == rhs.c.y))
    expected = select(list(unique(expected.columns, key=lambda c:
                                  c.name))).select_from(expected)

    result = compute(joined, {L: lhs, R: rhs}, return_type='native')

    assert str(result) == str(expected)

    assert str(select(result)) == str(select(expected))

    # Schemas match
    print(result.c.keys())
    print(joined.fields)
    assert list(result.c.keys()) == list(joined.fields)


def test_unary_op():
    assert str(compute(exp(t['amount']), s, post_compute=False, return_type='native')) == \
        str(sa.func.exp(s.c.amount))

    assert str(compute(-t['amount'], s, post_compute=False, return_type='native')) == \
        str(-s.c.amount)


@pytest.mark.parametrize('unbiased', [True, False])
def test_std(unbiased):
    assert str(compute(t.amount.std(unbiased=unbiased), s, post_compute=False, return_type='native')) == \
        str(getattr(sa.func,
                    'stddev_%s' % ('samp' if unbiased else 'pop'))(s.c.amount))


@pytest.mark.parametrize('unbiased', [True, False])
def test_var(unbiased):
    assert str(compute(t.amount.var(unbiased=unbiased), s, post_compute=False, return_type='native')) == \
        str(getattr(sa.func,
                    'var_%s' % ('samp' if unbiased else 'pop'))(s.c.amount))


def test_reductions():
    assert str(compute(sum(t['amount']), s, post_compute=False, return_type='native')) == \
        str(sa.sql.functions.sum(s.c.amount))
    assert str(compute(mean(t['amount']), s, post_compute=False, return_type='native')) == \
        str(sa.sql.func.avg(s.c.amount))
    assert str(compute(count(t['amount']), s, post_compute=False, return_type='native')) == \
        str(sa.sql.func.count(s.c.amount))

    assert 'amount_sum' == compute(
        sum(t['amount']), s, post_compute=False, return_type='native').name


def test_reduction_with_invalid_axis_argument():
    with pytest.raises(ValueError):
        compute(t.amount.mean(axis=1))

    with pytest.raises(ValueError):
        compute(t.count(axis=1))

    with pytest.raises(ValueError):
        compute(t[['amount', 'id']].count(axis=1))


def test_nelements():
    rhs = str(compute(t.count(), s, return_type='native'))
    assert str(compute(t.nelements(), s, return_type='native')) == rhs
    assert str(compute(t.nelements(axis=None), s, return_type='native')) == rhs
    assert str(compute(t.nelements(axis=0), s, return_type='native')) == rhs
    assert str(compute(t.nelements(axis=(0,)), s, return_type='native')) == rhs


@pytest.mark.xfail(raises=Exception, reason="We don't support axis=1 for"
                   " Record datashapes")
def test_nelements_axis_1():
    assert compute(t.nelements(axis=1), s, return_type='native') == len(s.columns)


def test_count_on_table():
    result = compute(t.count(), s, return_type='native')
    assert normalize(str(result)) == normalize("""
    SELECT count(accounts.id) as count_1
    FROM accounts""")

    result = compute(t[t.amount > 0].count(), s, return_type='native')
    assert (
        normalize(str(result)) == normalize("""
        SELECT count(accounts.id) as t_count
        FROM accounts
        WHERE accounts.amount > :amount_1""")

        or

        normalize(str(result)) == normalize("""
        SELECT count(alias.id) as t_count
        FROM (SELECT accounts.name AS name, accounts.amount AS amount, accounts.id AS id
              FROM accounts
              WHERE accounts.amount > :amount_1) as alias_2"""))


def test_distinct():
    result = str(compute(t['amount'].distinct(), s, post_compute=False, return_type='native'))

    assert 'distinct' in result.lower()
    assert 'amount' in result.lower()

    print(result)
    assert result == str(sa.distinct(s.c.amount))


def test_distinct_multiple_columns():
    assert normalize(str(compute(t.distinct(), s, return_type='native'))) == normalize("""
    SELECT DISTINCT accounts.name, accounts.amount, accounts.id
    FROM accounts""")


def test_nunique():
    result = str(computefull(nunique(t['amount']), s))

    print(result)
    assert 'distinct' in result.lower()
    assert 'count' in result.lower()
    assert 'amount' in result.lower()


def test_nunique_table():
    result = normalize(str(computefull(t.nunique(), s)))
    expected = normalize("""SELECT count(alias.id) AS tbl_row_count
FROM (SELECT DISTINCT accounts.name AS name, accounts.amount AS amount, accounts.id AS id
FROM accounts) as alias""")
    assert result == expected


@xfail(reason="Fails because SQLAlchemy doesn't seem to know binary reductions")
def test_binary_reductions():
    assert str(compute(any(t['amount'] > 150), s, return_type='native')) == \
        str(sa.sql.functions.any(s.c.amount > 150))


def test_by():
    expr = by(t['name'], total=t['amount'].sum())
    result = compute(expr, s, return_type='native')
    expected = sa.select([s.c.name,
                          sa.sql.functions.sum(s.c.amount).label('total')]
                         ).group_by(s.c.name)

    assert str(result) == str(expected)


def test_by_head():
    t2 = t.head(100)
    expr = by(t2['name'], total=t2['amount'].sum())
    result = compute(expr, s, return_type='native')
    expected = """
    SELECT accounts.name, sum(accounts.amount) as total
    FROM accounts
    GROUP by accounts.name
    LIMIT :param_1"""

    assert normalize(str(result)) == normalize(str(expected))


def test_by_two():
    expr = by(tbig[['name', 'sex']], total=tbig['amount'].sum())
    result = compute(expr, sbig, return_type='native')
    expected = (sa.select([sbig.c.name,
                           sbig.c.sex,
                           sa.sql.functions.sum(sbig.c.amount).label('total')])
                .group_by(sbig.c.name, sbig.c.sex))

    assert str(result) == str(expected)


def test_by_three():
    result = compute(by(tbig[['name', 'sex']],
                        total=(tbig['id'] + tbig['amount']).sum()),
                     sbig, return_type='native')

    assert normalize(str(result)) == normalize("""
    SELECT accountsbig.name,
           accountsbig.sex,
           sum(accountsbig.id + accountsbig.amount) AS total
    FROM accountsbig GROUP BY accountsbig.name, accountsbig.sex
    """)


def test_by_summary_clean():
    expr = by(t.name, min=t.amount.min(), max=t.amount.max())
    result = compute(expr, s, return_type='native')

    expected = """
    SELECT accounts.name, max(accounts.amount) AS max, min(accounts.amount) AS min
    FROM accounts
    GROUP BY accounts.name
    """

    assert normalize(str(result)) == normalize(expected)


def test_by_summary_single_column():
    expr = by(t.name, n=t.name.count(), biggest=t.name.max())
    result = compute(expr, s, return_type='native')

    expected = """
    SELECT accounts.name, max(accounts.name) AS biggest, count(accounts.name) AS n
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

    L = symbol('L', 'var * {name: string, amount: int}')
    R = symbol('R', 'var * {name: string, id: int}')
    want = join(L, R, 'name')[['amount', 'id']]

    result = compute(want, {L: lhs, R: rhs}, return_type='native')
    print(result)
    assert 'join' in str(result).lower()
    assert result.c.keys() == ['amount', 'id']
    assert 'amounts.name = ids.name' in str(result)


def test_sort():
    assert str(compute(t.sort('amount'), s, return_type='native')) == \
        str(select(s).order_by(sa.asc(s.c.amount)))

    assert str(compute(t.sort('amount', ascending=False), s, return_type='native')) == \
        str(select(s).order_by(sa.desc(s.c.amount)))


def test_multicolumn_sort():
    assert str(compute(t.sort(['amount', 'id']), s, return_type='native')) == \
        str(select(s).order_by(sa.asc(s.c.amount), sa.asc(s.c.id)))

    assert str(compute(t.sort(['amount', 'id'], ascending=False), s, return_type='native')) == \
        str(select(s).order_by(sa.desc(s.c.amount), sa.desc(s.c.id)))


def test_sort_on_distinct():
    assert normalize(str(compute(t.amount.sort(), s, return_type='native'))) == normalize("""
            SELECT accounts.amount
            FROM accounts
            ORDER BY accounts.amount ASC""")

    assert normalize(str(compute(t.amount.distinct().sort(), s, return_type='native'))) == normalize("""
            SELECT DISTINCT accounts.amount as amount
            FROM accounts
            ORDER BY amount ASC""")


def test_head():
    assert str(compute(t.head(2), s, return_type='native')) == str(select(s).limit(2))


def test_sample():
    order_by = select(s).order_by(sa.func.random())
    assert str(compute(t.sample(n=5), s)) == str(order_by.limit(5))


def test_label():
    assert (str(compute((t['amount'] * 10).label('foo'),
                        s, post_compute=False, return_type='native')) ==
            str((s.c.amount * 10).label('foo')))


def test_relabel_table():
    result = compute(t.relabel(name='NAME', id='ID'), s, return_type='native')
    expected = select([
        s.c.name.label('NAME'),
        s.c.amount,
        s.c.id.label('ID'),
    ])

    assert str(result) == str(expected)


def test_relabel_columns_over_selection():
    result = compute(
        t[t['amount'] > 150].relabel(name='NAME'), s, return_type='native'
    )
    expected = sa.select(
        [s.c.name.label('NAME'), s.c.amount, s.c.id]
    ).where(s.c.amount > 150)
    assert literal_compile(result) == literal_compile(expected)


def test_relabel_projection():
    result = compute(
        t[['name', 'id']].relabel(name='new_name', id='new_id'),
        s,
        return_type='native'
    )
    assert normalize(str(result)) == normalize(
        """SELECT
            accounts.name AS new_name,
            accounts.id AS new_id
        FROM accounts""",
    )


def test_merge():
    col = (t['amount'] * 2).label('new')

    expr = merge(t['name'], col)

    result = str(compute(expr, s, return_type='native'))

    assert 'amount * ' in result
    assert 'FROM accounts' in result
    assert 'SELECT accounts.name' in result
    assert 'new' in result


def test_projection_of_selection():
    print(compute(t[t['amount'] < 0][['name', 'amount']], s, return_type='native'))
    assert len(str(compute(t[t['amount'] < 0], s, return_type='native'))) > \
        len(str(compute(t[t['amount'] < 0][['name', 'amount']], s, return_type='native')))


def test_outer_join():
    L = symbol('L', 'var * {id: int, name: string, amount: real}')
    R = symbol('R', 'var * {city: string, id: int}')

    with tmpfile('db') as fn:
        uri = 'sqlite:///' + fn
        engine = bz_data(uri).data

        _left = [(1, 'Alice', 100),
                 (2, 'Bob', 200),
                 (4, 'Dennis', 400)]

        left = bz_data(uri + '::left', dshape=L.dshape)
        into(left, _left)

        _right = [('NYC', 1),
                  ('Boston', 1),
                  ('LA', 3),
                  ('Moscow', 4)]
        right = bz_data(uri + '::right', dshape=R.dshape)
        into(right, _right)

        conn = engine.connect()

        query = compute(join(L, R, how='inner'),
                        {L: left, R: right},
                        post_compute=False, return_type='native')
        result = list(map(tuple, conn.execute(query).fetchall()))

        assert set(result) == set(
            [(1, 'Alice', 100, 'NYC'),
             (1, 'Alice', 100, 'Boston'),
             (4, 'Dennis', 400, 'Moscow')])

        query = compute(join(L, R, how='left'),
                        {L: left, R: right},
                        post_compute=False, return_type='native')
        result = list(map(tuple, conn.execute(query).fetchall()))

        assert set(result) == set(
            [(1, 'Alice', 100, 'NYC'),
             (1, 'Alice', 100, 'Boston'),
             (2, 'Bob', 200, None),
             (4, 'Dennis', 400, 'Moscow')])

        query = compute(join(L, R, how='right'),
                        {L: left, R: right},
                        post_compute=False, return_type='native')
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
        query = compute(join(L, R, how='outer'),
                        {L: left, R: right},
                        post_compute=False)
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
    result = str(compute(expr, s, return_type='native'))

    assert 'sum(accounts.amount) as a' in result.lower()
    assert 'count(accounts.id) as b' in result.lower()


def test_summary_clean():
    t2 = t[t.amount > 0]
    expr = summary(a=t2.amount.sum(), b=t2.id.count())
    result = str(compute(expr, s, return_type='native'))

    assert normalize(result) == normalize("""
    SELECT sum(accounts.amount) as a, count(accounts.id) as b
    FROM accounts
    WHERE accounts.amount > :amount_1""")


def test_summary_by():
    expr = by(t.name, summary(a=t.amount.sum(), b=t.id.count()))

    result = str(compute(expr, s, return_type='native'))

    assert 'sum(accounts.amount) as a' in result.lower()
    assert 'count(accounts.id) as b' in result.lower()

    assert 'group by accounts.name' in result.lower()


def test_clean_join():
    metadata = sa.MetaData()
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

    tcity = symbol('city', discover(city))
    tfriends = symbol('friends', discover(friends))
    tname = symbol('name', discover(name))

    ns = {tname: name, tfriends: friends, tcity: city}

    expr = join(tfriends, tname, 'a', 'id')
    assert normalize(str(compute(expr, ns, return_type='native'))) == normalize("""
    SELECT friends.a, friends.b, name.name
    FROM friends JOIN name on friends.a = name.id""")

    expr = join(join(tfriends, tname, 'a', 'id'), tcity, 'a', 'id')

    result = compute(expr, ns, return_type='native')

    expected1 = """
    SELECT friends.a, friends.b, name.name, place.city, place.country
    FROM friends
        JOIN name ON friends.a = name.id
        JOIN place ON friends.a = place.id
        """

    expected2 = """
    SELECT alias.a, alias.b, alias.name, place.city, place.country
    FROM (SELECT friends.a AS a, friends.b AS b, name.name AS name
          FROM friends JOIN name ON friends.a = name.id) AS alias
    JOIN place ON alias.a = place.id
    """

    assert (normalize(str(result)) == normalize(expected1) or
            normalize(str(result)) == normalize(expected2))


def test_like():
    expr = t[t.name.like('Alice*')]
    assert normalize(str(compute(expr, s, return_type='native'))) == normalize("""
    SELECT accounts.name, accounts.amount, accounts.id
    FROM accounts
    WHERE accounts.name LIKE :name_1""")

def test_like_join():
    # Test .like() on a non-trivial SELECT query.
    metadata = sa.MetaData()
    t0 = sa.Table('t0', metadata,
                   sa.Column('a', sa.String),
                   sa.Column('y', sa.Integer))

    t1 = sa.Table('t1', metadata,
                  sa.Column('b', sa.String),
                  sa.Column('z', sa.Integer))

    # expected = t0.join(t1, t0.c.a == t1.c.b)
    # expected = sa.select(list(unique(expected.columns, key=lambda c:
                                  # c.name))).select_from(expected)
    # expected = expected.where(expected.c.a.like('%a'))

    t0sym = symbol('t0sym', 'var * {a: string, y: int}')
    t1sym = symbol('t1sym', 'var * {b: string, z: int}')
    joined = join(t0sym, t1sym, on_left='a', on_right='b')
    liked = joined[joined.a.like('*a')]
    result = compute(liked, {t0sym: t0, t1sym: t1}, return_type='native')
    assert normalize(str(result)) == normalize("""
    SELECT t0.a, t0.y, t1.z
    FROM t0 JOIN t1 ON t0.a = t1.b
    WHERE t0.a LIKE :a_1""")


def test_not_like():
    expr = t[~t.name.like('Alice*')]
    assert normalize(str(compute(expr, s, return_type='native'))) == normalize("""
    SELECT accounts.name, accounts.amount, accounts.id
    FROM accounts
    WHERE accounts.name NOT LIKE :name_1""")


def test_str_len():
    expr = t.name.str_len()
    result = str(compute(expr, s, return_type='native'))
    expected = "SELECT char_length(accounts.name) as name FROM accounts"
    assert normalize(result) == normalize(expected)


def test_str_upper():
    expr = t.name.str_upper()
    result = str(compute(expr, s, return_type='native'))
    expected = "SELECT upper(accounts.name) as name FROM accounts"
    assert normalize(result) == normalize(expected)


def test_str_lower():
    expr = t.name.str_lower()
    result = str(compute(expr, s, return_type='native'))
    expected = "SELECT lower(accounts.name) as name FROM accounts"
    assert normalize(result) == normalize(expected)


@pytest.mark.parametrize("sep", [None, " sep "])
def test_str_cat(sep):
    """
    Need at least two string columns to test str_cat
    """

    if sep is None:
        expr = t_str_cat.name.str_cat(t_str_cat.comment)
        expected = """
                   SELECT accounts2.name || accounts2.comment
                   AS anon_1 FROM accounts2
                   """
    else:
        expr = t_str_cat.name.str_cat(t_str_cat.comment, sep=sep)
        expected = """
                   SELECT accounts2.name || :name_1 || accounts2.comment
                   AS anon_1 FROM accounts2
                   """

    result = str(compute(expr, s_str_cat, return_type='native'))
    assert normalize(result) == normalize(expected)


def test_str_cat_chain():
    expr = (t_str_cat.name
            .str_cat(t_str_cat.comment, sep=' -- ')
            .str_cat(t_str_cat.product, sep=' ++ '))
    result = str(compute(expr, {t_str_cat: s_str_cat}, return_type='native'))
    expected = """
               SELECT accounts2.name || :name_1 || accounts2.comment ||
               :param_1 || accounts2.product AS anon_1 FROM accounts2
               """
    assert normalize(result) == normalize(expected)


def test_str_cat_no_runtime_exception():
    """
    No exception raised if resource is the same
    """
    expr = t_str_cat.comment.str_cat(t.name)
    compute(expr, {t: s_str_cat, t_str_cat: s_str_cat}, return_type='native')


def test_columnwise_on_complex_selection():
    result = str(select(compute(t[t.amount > 0].amount + 1, s, return_type='native')))
    expected = """
               SELECT accounts.amount + :amount_1 AS amount
               FROM accounts
               WHERE accounts.amount > :amount_2
               """
    assert normalize(result) == normalize(expected)


def test_reductions_on_complex_selections():
    assert (
        normalize(str(select(compute(t[t.amount > 0].id.sum(), s, return_type='native')))) ==
        normalize(
            """
            select
                sum(alias.id) as id_sum
            from (select
                    accounts.id as id
                  from accounts
                  where accounts.amount > :amount_1) as alias
            """
        )
    )


def test_clean_summary_by_where():
    t2 = t[t.id == 1]
    expr = by(t2.name, sum=t2.amount.sum(), count=t2.amount.count())
    result = compute(expr, s, return_type='native')

    assert normalize(str(result)) == normalize("""
    SELECT accounts.name, count(accounts.amount) AS count, sum(accounts.amount) AS sum
    FROM accounts
    WHERE accounts.id = :id_1
    GROUP BY accounts.name
    """)


def test_by_on_count():
    expr = by(t.name, count=t.count())
    result = compute(expr, s, return_type='native')

    assert normalize(str(result)) == normalize("""
    SELECT accounts.name, count(accounts.id) AS count
    FROM accounts
    GROUP BY accounts.name
    """)


def test_join_complex_clean():
    metadata = sa.MetaData()
    name = sa.Table('name', metadata,
                    sa.Column('id', sa.Integer),
                    sa.Column('name', sa.String),
                    )
    city = sa.Table('place', metadata,
                    sa.Column('id', sa.Integer),
                    sa.Column('city', sa.String),
                    sa.Column('country', sa.String),
                    )

    tname = symbol('name', discover(name))
    tcity = symbol('city', discover(city))

    ns = {tname: name, tcity: city}

    expr = join(tname[tname.id > 0], tcity, 'id')
    result = compute(expr, ns, return_type='native')

    expected1 = """
        SELECT name.id, name.name, place.city, place.country
        FROM name JOIN place ON name.id = place.id
        WHERE name.id > :id_1"""

    expected2 = """
        SELECT alias.id, alias.name, place.city, place.country
        FROM (SELECT name.id as id, name.name AS name
              FROM name
              WHERE name.id > :id_1) AS alias
        JOIN place ON alias.id = place.id"""
    assert (normalize(str(result)) == normalize(expected1) or
            normalize(str(result)) == normalize(expected2))


def test_projection_of_join():
    metadata = sa.MetaData()
    name = sa.Table('name', metadata,
                    sa.Column('id', sa.Integer),
                    sa.Column('name', sa.String),
                    )
    city = sa.Table('place', metadata,
                    sa.Column('id', sa.Integer),
                    sa.Column('city', sa.String),
                    sa.Column('country', sa.String),
                    )

    tname = symbol('name', discover(name))
    tcity = symbol('city', discover(city))

    expr = join(tname, tcity[tcity.city == 'NYC'], 'id')[['country', 'name']]

    ns = {tname: name, tcity: city}

    result = compute(expr, ns, return_type='native')

    expected1 = """
        SELECT place.country, name.name
        FROM name JOIN place ON name.id = place.id
        WHERE place.city = :city_1"""

    expected2 = """
        SELECT alias.country, name.name
        FROM name
        JOIN (SELECT place.id AS id, place.city AS city, place.country AS country
              FROM place
              WHERE place.city = :city_1) AS alias
        ON name.id = alias_6.id"""

    assert (normalize(str(result)) == normalize(expected1) or
            normalize(str(result)) == normalize(expected2))


def test_lower_column():
    metadata = sa.MetaData()
    name = sa.Table('name', metadata,
                    sa.Column('id', sa.Integer),
                    sa.Column('name', sa.String),
                    )
    city = sa.Table('place', metadata,
                    sa.Column('id', sa.Integer),
                    sa.Column('city', sa.String),
                    sa.Column('country', sa.String),
                    )

    tname = symbol('name', discover(name))
    tcity = symbol('city', discover(city))

    assert lower_column(name.c.id) is name.c.id
    assert lower_column(select(name).c.id) is name.c.id

    j = name.join(city, name.c.id == city.c.id)
    col = [c for c in j.columns if c.name == 'country'][0]

    assert lower_column(col) is city.c.country


def test_selection_of_join():
    metadata = sa.MetaData()
    name = sa.Table('name', metadata,
                    sa.Column('id', sa.Integer),
                    sa.Column('name', sa.String),
                    )
    city = sa.Table('place', metadata,
                    sa.Column('id', sa.Integer),
                    sa.Column('city', sa.String),
                    sa.Column('country', sa.String),
                    )

    tname = symbol('name', discover(name))
    tcity = symbol('city', discover(city))

    ns = {tname: name, tcity: city}

    j = join(tname, tcity, 'id')
    expr = j[j.city == 'NYC'].name
    result = compute(expr, ns, return_type='native')

    assert normalize(str(result)) == normalize("""
    SELECT name.name
    FROM name JOIN place ON name.id = place.id
    WHERE place.city = :city_1""")


def test_join_on_same_table():
    metadata = sa.MetaData()
    T = sa.Table('tab', metadata,
                 sa.Column('a', sa.Integer),
                 sa.Column('b', sa.Integer),
                 )

    t = symbol('tab', discover(T))
    expr = join(t, t, 'a')

    result = compute(expr, {t: T}, return_type='native')

    assert normalize(str(result)) == normalize("""
    SELECT
        tab_left.a,
        tab_left.b as b_left,
        tab_right.b as b_right
    FROM
        tab AS tab_left
    JOIN
        tab AS tab_right
    ON
        tab_left.a = tab_right.a
    """)

    expr = join(t, t, 'a').b_left.sum()

    result = compute(expr, {t: T}, return_type='native')

    assert normalize(str(result)) == normalize("""
    select sum(alias.b_left) as b_left_sum from
        (select
            tab_left.b as b_left
        from
            tab as tab_left
        join
            tab as tab_right
        on
            tab_left.a = tab_right.a) as
        alias""")

    expr = join(t, t, 'a')
    expr = summary(total=expr.a.sum(), smallest=expr.b_right.min())

    result = compute(expr, {t: T}, return_type='native')

    assert normalize(str(result)) == normalize("""
    SELECT
        min(tab_right.b) as smallest,
        sum(tab_left.a) as total
    FROM
        tab AS tab_left
    JOIN
        tab AS tab_right
    ON
        tab_left.a = tab_right.a
    """)


def test_join_suffixes():
    metadata = sa.MetaData()
    T = sa.Table('tab', metadata,
                 sa.Column('a', sa.Integer),
                 sa.Column('b', sa.Integer),
                 )

    t = symbol('tab', discover(T))
    suffixes = '_l', '_r'
    expr = join(t, t, 'a', suffixes=suffixes)

    result = compute(expr, {t: T}, return_type='native')

    assert normalize(str(result)) == normalize("""
    SELECT
        tab{l}.a,
        tab{l}.b as b{l},
        tab{r}.b as b{r}
    FROM
        tab AS tab{l}
    JOIN
        tab AS tab{r}
    ON
        tab{l}.a = tab{r}.a
    """.format(l=suffixes[0], r=suffixes[1]))


def test_field_access_on_engines(city_data):
    s, engine = city_data['s'], city_data['engine']
    result = compute_up(s.city, engine, return_type='native')
    assert isinstance(result, sa.Table)
    assert result.name == 'city'


def test_computation_directly_on_sqlalchemy_Tables(city_data):
    name = city_data['name']
    s = symbol('s', discover(name))
    result = into(list, compute(s.id + 1, name, return_type='native'))
    assert not isinstance(result, sa.sql.Selectable)
    assert list(result) == []


def test_computation_directly_on_metadata(city_data):
    metadata = city_data['metadata']
    name = city_data['name']
    s = symbol('s', discover(metadata))
    result = compute(s.name, {s: metadata}, post_compute=False, return_type='native')
    assert result == name


sql_bank = sa.Table('bank', sa.MetaData(),
                    sa.Column('id', sa.Integer),
                    sa.Column('name', sa.String),
                    sa.Column('amount', sa.Integer))
sql_cities = sa.Table('cities', sa.MetaData(),
                      sa.Column('name', sa.String),
                      sa.Column('city', sa.String))

bank = symbol('bank', discover(sql_bank))
cities = symbol('cities', discover(sql_cities))


def test_aliased_views_with_two_group_bys():
    expr = by(bank.name, total=bank.amount.sum())
    expr2 = by(expr.total, count=expr.name.count())

    result = compute(expr2, {bank: sql_bank, cities: sql_cities}, return_type='native')

    assert normalize(str(result)) == normalize("""
    SELECT alias.total, count(alias.name) as count
    FROM (SELECT bank.name AS name, sum(bank.amount) AS total
          FROM bank
          GROUP BY bank.name) as alias
    GROUP BY alias.total
    """)


def test_aliased_views_with_join():
    joined = join(bank, cities)
    expr = by(joined.city, total=joined.amount.sum())
    expr2 = by(expr.total, count=expr.city.nunique())

    result = compute(expr2, {bank: sql_bank, cities: sql_cities}, return_type='native')

    assert normalize(str(result)) == normalize("""
    SELECT alias.total, count(DISTINCT alias.city) AS count
    FROM (SELECT cities.city AS city, sum(bank.amount) AS total
          FROM bank
          JOIN cities ON bank.name = cities.name
          GROUP BY cities.city) as alias
    GROUP BY alias.total
    """)


def test_select_field_on_alias():
    result = compute_up(t.amount, select(s).limit(10).alias('foo'))
    assert normalize(str(select(result))) == normalize("""
        SELECT foo.amount
        FROM (SELECT accounts.name AS name, accounts.amount AS amount, accounts.id AS id
              FROM accounts
              LIMIT :param_1) as foo""")


@pytest.mark.xfail(raises=Exception,
                   reason="sqlalchemy.join seems to drop unnecessary tables")
def test_join_on_single_column():
    expr = join(cities[['name']], bank)
    result = compute(expr, {bank: sql_bank, cities: sql_cities}, return_type='native')

    assert normalize(str(result)) == """
    SELECT bank.id, bank.name, bank.amount
    FROM bank join cities ON bank.name = cities.name"""

    expr = join(bank, cities.name)
    result = compute(expr, {bank: sql_bank, cities: sql_cities}, return_type='native')

    assert normalize(str(result)) == """
    SELECT bank.id, bank.name, bank.amount
    FROM bank join cities ON bank.name = cities.name"""


def test_aliased_views_more():
    metadata = sa.MetaData()
    lhs = sa.Table('aaa', metadata,
                   sa.Column('x', sa.Integer),
                   sa.Column('y', sa.Integer),
                   sa.Column('z', sa.Integer))

    rhs = sa.Table('bbb', metadata,
                   sa.Column('w', sa.Integer),
                   sa.Column('x', sa.Integer),
                   sa.Column('y', sa.Integer))

    L = symbol('L', 'var * {x: int, y: int, z: int}')
    R = symbol('R', 'var * {w: int, x: int, y: int}')

    expr = join(by(L.x, y_total=L.y.sum()),
                R)

    result = compute(expr, {L: lhs, R: rhs}, return_type='native')

    assert normalize(str(result)) == normalize("""
        SELECT alias.x, alias.y_total, bbb.w, bbb.y
        FROM (SELECT aaa.x as x, sum(aaa.y) as y_total
              FROM aaa
              GROUP BY aaa.x) AS alias
        JOIN bbb ON alias.x = bbb.x """)

    expr2 = by(expr.w, count=expr.x.count(), total2=expr.y_total.sum())

    result2 = compute(expr2, {L: lhs, R: rhs}, return_type='native')

    assert (
        normalize(str(result2)) == normalize("""
            SELECT alias_2.w, count(alias_2.x) as count, sum(alias_2.y_total) as total2
            FROM (SELECT alias.x, alias.y_total, bbb.w, bbb.y
                  FROM (SELECT aaa.x as x, sum(aaa.y) as y_total
                        FROM aaa
                        GROUP BY aaa.x) AS alias
                  JOIN bbb ON alias.x = bbb.x) AS alias_2
            GROUP BY alias_2.w""")

        or

        normalize(str(result2)) == normalize("""
            SELECT bbb.w, count(alias.x) as count, sum(alias.y_total) as total2
            FROM (SELECT aaa.x as x, sum(aaa.y) as y_total
                  FROM aaa
                  GROUP BY aaa.x) as alias
              JOIN bbb ON alias.x = bbb.x
            GROUP BY bbb.w"""))


def test_aliased_views_with_computation():
    engine = sa.create_engine('sqlite:///:memory:')

    df_aaa = DataFrame({'x': [1, 2, 3, 2, 3],
                        'y': [2, 1, 2, 3, 1],
                        'z': [3, 3, 3, 1, 2]})
    df_bbb = DataFrame({'w': [1, 2, 3, 2, 3],
                        'x': [2, 1, 2, 3, 1],
                        'y': [3, 3, 3, 1, 2]})

    df_aaa.to_sql('aaa', engine)
    df_bbb.to_sql('bbb', engine)

    metadata = sa.MetaData(engine)
    metadata.reflect()

    sql_aaa = metadata.tables['aaa']
    sql_bbb = metadata.tables['bbb']

    L = symbol('aaa', discover(df_aaa))
    R = symbol('bbb', discover(df_bbb))

    expr = join(by(L.x, y_total=L.y.sum()),
                R)
    a = compute(expr, {L: df_aaa, R: df_bbb}, return_type='native')
    b = compute(expr, {L: sql_aaa, R: sql_bbb}, return_type='native')
    assert into(set, a) == into(set, b)

    expr2 = by(expr.w, count=expr.x.count(), total2=expr.y_total.sum())
    a = compute(expr2, {L: df_aaa, R: df_bbb}, return_type='native')
    b = compute(expr2, {L: sql_aaa, R: sql_bbb}, return_type='native')
    assert into(set, a) == into(set, b)

    expr3 = by(expr.x, count=expr.y_total.count())
    a = compute(expr3, {L: df_aaa, R: df_bbb}, return_type='native')
    b = compute(expr3, {L: sql_aaa, R: sql_bbb}, return_type='native')
    assert into(set, a) == into(set, b)

    expr4 = join(expr2, R)
    a = compute(expr4, {L: df_aaa, R: df_bbb}, return_type='native')
    b = compute(expr4, {L: sql_aaa, R: sql_bbb}, return_type='native')
    assert into(set, a) == into(set, b)

    """ # Takes a while
    expr5 = by(expr4.count, total=(expr4.x + expr4.y).sum())
    a = compute(expr5, {L: df_aaa, R: df_bbb})
    b = compute(expr5, {L: sql_aaa, R: sql_bbb})
    assert into(set, a) == into(set, b)
    """


def test_distinct_count_on_projection():
    expr = t[['amount']].distinct().count()

    result = compute(expr, {t: s}, return_type='native')

    assert (
        normalize(str(result)) == normalize("""
        SELECT count(DISTINCT accounts.amount)
        FROM accounts""")

        or

        normalize(str(result)) == normalize("""
        SELECT count(alias.amount) as count
        FROM (SELECT DISTINCT accounts.amount AS amount
              FROM accounts) as alias"""))

    # note that id is the primary key
    expr = t[['amount', 'id']].distinct().count()

    result = compute(expr, {t: s}, return_type='native')
    assert normalize(str(result)) == normalize("""
        SELECT count(alias.id) as count
        FROM (SELECT DISTINCT accounts.amount AS amount, accounts.id AS id
              FROM accounts) as alias""")


def test_join_count():
    ds = datashape.dshape(
        '{t1: var * {x: int, y: int}, t2: var * {a: int, b: int}}')
    engine = bz_data('sqlite:///:memory:', dshape=ds)
    db = symbol('db', ds)

    expr = join(db.t1[db.t1.x > -1], db.t2, 'x', 'a').count()

    result = compute(expr, {db: engine}, post_compute=False, return_type='native')

    expected1 = """
    SELECT count(alias.x) as count
    FROM (SELECT t1.x AS x, t1.y AS y, t2.b AS b
          FROM t1 JOIN t2 ON t1.x = t2.a
          WHERE t1.x > ?) as alias
          """
    expected2 = """
    SELECT count(alias2.x) AS count
    FROM (SELECT alias1.x AS x, alias1.y AS y, t2.b AS b
          FROM (SELECT t1.x AS x, t1.y AS y
                FROM t1
                WHERE t1.x > ?) AS alias1
          JOIN t2 ON alias1.x = t2.a) AS alias2"""

    assert (normalize(str(result)) == normalize(expected1) or
            normalize(str(result)) == normalize(expected2))


def test_transform_where():
    t2 = t[t.id == 1]
    expr = transform(t2, abs_amt=abs(t2.amount), sine=sin(t2.id))
    result = compute(expr, s, return_type='native')

    expected = """SELECT
        accounts.name,
        accounts.amount,
        accounts.id,
        abs(accounts.amount) as abs_amt,
        sin(accounts.id) as sine
    FROM accounts
    WHERE accounts.id = :id_1
    """

    assert normalize(str(result)) == normalize(expected)


def test_merge():
    col = (t['amount'] * 2).label('new')

    expr = merge(t['name'], col)

    result = str(compute(expr, s, return_type='native'))

    assert 'amount * ' in result
    assert 'FROM accounts' in result
    assert 'SELECT accounts.name' in result
    assert 'new' in result


def test_merge_where():
    t2 = t[t.id == 1]
    expr = merge(t2[['amount', 'name']], t2.id)
    result = compute(expr, s, return_type='native')
    expected = normalize("""SELECT
        accounts.amount,
        accounts.name,
        accounts.id
    FROM accounts
    WHERE accounts.id = :id_1
    """)
    assert normalize(str(result)) == expected


def test_transform_filter_by_single_column():
    t2 = t[t.amount < 0]
    tr = transform(t2, abs_amt=abs(t2.amount), sine=sin(t2.id))
    expr = by(tr.name, avg_amt=tr.abs_amt.mean())
    result = compute(expr, s, return_type='native')
    expected = normalize("""SELECT
        accounts.name,
        avg(abs(accounts.amount)) AS avg_amt
    FROM accounts
    WHERE accounts.amount < :amount_1
    GROUP BY accounts.name
    """)
    assert normalize(str(result)) == expected


def test_transform_filter_by_multiple_columns():
    t2 = t[t.amount < 0]
    tr = transform(t2, abs_amt=abs(t2.amount), sine=sin(t2.id))
    expr = by(tr.name, avg_amt=tr.abs_amt.mean(), sum_sine=tr.sine.sum())
    result = compute(expr, s, return_type='native')
    expected = normalize("""SELECT
        accounts.name,
        avg(abs(accounts.amount)) AS avg_amt,
        sum(sin(accounts.id)) AS sum_sine
    FROM accounts
    WHERE accounts.amount < :amount_1
    GROUP BY accounts.name
    """)
    assert normalize(str(result)) == expected


def test_transform_filter_by_different_order():
    t2 = transform(t, abs_amt=abs(t.amount), sine=sin(t.id))
    tr = t2[t2.amount < 0]
    expr = by(tr.name,
              avg_amt=tr.abs_amt.mean(),
              avg_sine=tr.sine.sum() / tr.sine.count())
    result = compute(expr, s, return_type='native')
    expected = normalize("""SELECT
        accounts.name,
        avg(abs(accounts.amount)) AS avg_amt,
        sum(sin(accounts.id)) / count(sin(accounts.id)) AS avg_sine
    FROM accounts
    WHERE accounts.amount < :amount_1
    GROUP BY accounts.name
    """)
    assert normalize(str(result)) == expected


def test_transform_filter_by_projection():
    t2 = transform(t, abs_amt=abs(t.amount), sine=sin(t.id))
    tr = t2[t2.amount < 0]
    expr = by(tr[['name', 'id']],
              avg_amt=tr.abs_amt.mean(),
              avg_sine=tr.sine.sum() / tr.sine.count())
    result = compute(expr, s, return_type='native')
    expected = normalize("""SELECT
        accounts.name,
        accounts.id,
        avg(abs(accounts.amount)) AS avg_amt,
        sum(sin(accounts.id)) / count(sin(accounts.id)) AS avg_sine
    FROM accounts
    WHERE accounts.amount < :amount_1
    GROUP BY accounts.name, accounts.id
    """)
    assert normalize(str(result)) == expected


def test_merge_compute():
    dta = [(1, 'Alice', 100),
            (2, 'Bob', 200),
            (4, 'Dennis', 400)]
    ds = datashape.dshape('var * {id: int, name: string, amount: real}')
    s = symbol('s', ds)

    with tmpfile('db') as fn:
        uri = 'sqlite:///' + fn
        into(uri + '::table', dta, dshape=ds)

        expr = transform(s, amount10=s.amount * 10)
        result = into(list, compute(expr, {s: dta}, return_type='native'))

        assert result == [(1, 'Alice', 100, 1000),
                          (2, 'Bob', 200, 2000),
                          (4, 'Dennis', 400, 4000)]


def test_notnull():
    result = compute(nt[nt.name.notnull()], ns, return_type='native')
    expected = """SELECT
        nullaccounts.name,
        nullaccounts.amount,
        nullaccounts.id
    FROM nullaccounts
    WHERE nullaccounts.name is not null
    """
    assert normalize(str(result)) == normalize(expected)


def test_head_limit():
    assert compute(t.head(5).head(10), s, return_type='native')._limit == 5
    assert compute(t.head(10).head(5), s, return_type='native')._limit == 5
    assert compute(t.head(10).head(10), s, return_type='native')._limit == 10


def test_no_extraneous_join():
    ds = """ {event: var * {name: ?string,
                            operation: ?string,
                            datetime_nearest_receiver: ?datetime,
                            aircraft: ?string,
                            temperature_2m: ?float64,
                            temperature_5cm: ?float64,
                            humidity: ?float64,
                            windspeed: ?float64,
                            pressure: ?float64,
                            include: int64},
             operation: var * {name: ?string,
                               runway: int64,
                               takeoff: bool,
                               datetime_nearest_close: ?string}}
        """
    db = bz_data('sqlite:///:memory:', dshape=ds)

    d = symbol('db', dshape=ds)

    expr = join(d.event[d.event.include == True],
                d.operation[['name', 'datetime_nearest_close']],
                'operation', 'name')

    result = compute(expr, db, return_type='native')

    assert normalize(str(result)) == normalize("""
    SELECT
        alias.operation,
        alias.name as name_left,
        alias.datetime_nearest_receiver,
        alias.aircraft,
        alias.temperature_2m,
        alias.temperature_5cm,
        alias.humidity,
        alias.windspeed,
        alias.pressure,
        alias.include,
        alias.datetime_nearest_close
    FROM
        (SELECT
             event.name AS name,
             event.operation AS operation,
             event.datetime_nearest_receiver AS datetime_nearest_receiver,
             event.aircraft AS aircraft,
             event.temperature_2m AS temperature_2m,
             event.temperature_5cm AS temperature_5cm,
             event.humidity AS humidity,
             event.windspeed AS windspeed,
             event.pressure AS pressure,
             event.include AS include
         FROM
             event WHERE event.include = 1) AS alias1
    JOIN
        (SELECT
             operation.name AS name,
             operation.datetime_nearest_close as datetime_nearest_close
         FROM operation) AS alias2
    ON
        alias1.operation = alias2.name
    """)


def test_math():
    result = compute(sin(t.amount), s, return_type='native')
    assert normalize(str(result)) == normalize("""
            SELECT sin(accounts.amount) as amount
            FROM accounts""")

    result = compute(floor(t.amount), s, return_type='native')
    assert normalize(str(result)) == normalize("""
            SELECT floor(accounts.amount) as amount
            FROM accounts""")

    result = compute(t.amount // 2, s, return_type='native')
    assert normalize(str(result)) == normalize("""
            SELECT floor(accounts.amount / :amount_1) AS amount
            FROM accounts""")


def test_transform_order():
    r = transform(t, sin_amount=sin(t.amount), cos_id=cos(t.id))
    result = compute(r, s, return_type='native')
    expected = """SELECT
        accounts.name,
        accounts.amount,
        accounts.id,
        cos(accounts.id) as cos_id,
        sin(accounts.amount) as sin_amount
    FROM accounts
    """
    assert normalize(str(result)) == normalize(expected)


def test_isin():
    result = t[t.name.isin(['foo', 'bar'])]
    result_sql_expr = str(compute(result, s, return_type='native'))
    expected = """
        SELECT
            accounts.name,
            accounts.amount,
            accounts.id
        FROM
            accounts
        WHERE
            accounts.name
        IN
            (:name_1,
            :name_2)
    """
    assert normalize(result_sql_expr) == normalize(expected)


@pytest.mark.skipif('1.0.0' <= LooseVersion(sa.__version__) <= '1.0.1',
                    reason=("SQLAlchemy generates different code in 1.0.0"
                            " and 1.0.1"))
def test_date_grouper_repeats_not_one_point_oh():
    columns = [sa.Column('amount', sa.REAL),
               sa.Column('ds', sa.TIMESTAMP)]
    dta = sa.Table('t', sa.MetaData(), *columns)
    t = symbol('t', discover(dta))
    expr = by(t.ds.year, avg_amt=t.amount.mean())
    result = str(compute(expr, dta, return_type='native'))

    # FYI spark sql isn't able to parse this correctly
    expected = """SELECT
        EXTRACT(year FROM t.ds) as ds_year,
        AVG(t.amount) as avg_amt
    FROM t
    GROUP BY EXTRACT(year FROM t.ds)
    """
    assert normalize(result) == normalize(expected)


@pytest.mark.skipif(LooseVersion(sa.__version__) < '1.0.0' or
                    LooseVersion(sa.__version__) >= '1.0.2',
                    reason=("SQLAlchemy generates different code in < 1.0.0 "
                            "and >= 1.0.2"))
def test_date_grouper_repeats():
    columns = [sa.Column('amount', sa.REAL),
               sa.Column('ds', sa.TIMESTAMP)]
    dta = sa.Table('t', sa.MetaData(), *columns)
    t = symbol('t', discover(dta))
    expr = by(t.ds.year, avg_amt=t.amount.mean())
    result = str(compute(expr, dta, return_type='native'))

    # FYI spark sql isn't able to parse this correctly
    expected = """SELECT
        EXTRACT(year FROM t.ds) as ds_year,
        AVG(t.amount) as avg_amt
    FROM t
    GROUP BY ds_year
    """
    assert normalize(result) == normalize(expected)


def test_transform_then_project_single_column():
    expr = transform(t, foo=t.id + 1)[['foo', 'id']]
    result = normalize(str(compute(expr, s, return_type='native')))
    expected = normalize("""SELECT
        accounts.id + :id_1 as foo,
        accounts.id
    FROM accounts""")
    assert result == expected


def test_transform_then_project():
    proj = ['foo', 'id']
    expr = transform(t, foo=t.id + 1)[proj]
    result = normalize(str(compute(expr, s, return_type='native')))
    expected = normalize("""SELECT
        accounts.id + :id_1 as foo,
        accounts.id
    FROM accounts""")
    assert result == expected


def test_reduce_does_not_compose():
    expr = by(t.name, counts=t.count()).counts.max()
    result = str(compute(expr, s, return_type='native'))
    expected = """
    SELECT max(alias.counts) AS counts_max
    FROM
    (SELECT count(accounts.id) AS counts
    FROM accounts GROUP BY accounts.name) as alias"""
    assert normalize(result) == normalize(expected)


@pytest.mark.xfail(raises=NotImplementedError)
def test_normalize_reduction():
    expr = by(t.name, counts=t.count())
    expr = transform(expr, normed_counts=expr.counts / expr.counts.max())
    result = str(compute(expr, s, return_type='native'))
    expected = """WITH alias AS
(SELECT count(accounts.id) AS counts
FROM accounts GROUP BY accounts.name)
 SELECT alias.counts / max(alias.counts) AS normed_counts
FROM alias"""
    assert normalize(result) == normalize(expected)


def test_do_not_erase_group_by_functions_with_datetime():
    t, s = tdate, sdate
    expr = by(t[t.amount < 0].occurred_on.date,
              avg_amount=t[t.amount < 0].amount.mean())
    result = str(compute(expr, s, return_type='native'))
    expected = """SELECT
        date(accdate.occurred_on) as occurred_on_date,
        avg(accdate.amount) as avg_amount
    FROM
        accdate
    WHERE
        accdate.amount < :amount_1
    GROUP BY
        date(accdate.occurred_on)
    """
    assert normalize(result) == normalize(expected)


def test_not():
    expr = t.amount[~t.name.isin(('Billy', 'Bob'))]
    result = str(compute(expr, s, return_type='native'))
    expected = """SELECT
        accounts.amount
    FROM
        accounts
    WHERE
        accounts.name not in (:name_1, :name_2)
    """
    assert normalize(result) == normalize(expected)


def test_slice():
    start, stop, step = 50, 100, 1
    result = str(compute(t[start:stop], s, return_type='native'))

    # Verifies that compute is translating the query correctly
    assert result == str(select(s).offset(start).limit(stop))

    # Verifies the query against expected SQL query
    expected = """
    SELECT accounts.name, accounts.amount, accounts.id FROM accounts
    LIMIT :param_1 OFFSET :param_2
    """

    assert normalize(str(result)) == normalize(str(expected))

    # Step size of 1 should be alright
    compute(t[start:stop:step], s, return_type='native')


@pytest.mark.xfail(raises=ValueError)
def test_slice_step():
    start, stop, step = 50, 100, 2
    compute(t[start:stop:step], s, return_type='native')


def test_datetime_to_date():
    expr = tdate.occurred_on.date
    result = str(compute(expr, sdate, return_type='native'))
    expected = """SELECT
        DATE(accdate.occurred_on) as occurred_on_date
    FROM
        accdate
    """
    assert normalize(result) == normalize(expected)


def test_sort_compose():
    expr = t.name[:5].sort()
    result = compute(expr, s, return_type='native')
    expected = """select
            anon_1.name
        from (select
                  accounts.name as name
              from
                  accounts
              limit :param_1
              offset :param_2) as anon_1
        order by
            anon_1.name asc"""
    assert normalize(str(result)) == normalize(expected)
    assert (normalize(str(compute(t.sort('name').name[:5], s, return_type='native'))) !=
            normalize(expected))


def test_coerce():
    expr = t.amount.coerce(to='int64')
    expected = """SELECT
        cast(accounts.amount AS BIGINT) AS amount
    FROM accounts"""
    result = compute(expr, s, return_type='native')
    assert normalize(str(result)) == normalize(expected)


def test_multi_column_by_after_transform():
    tbl = transform(t, new_amount=t.amount + 1, one_two=t.amount * 2)
    expr = by(tbl[['name', 'one_two']], avg_amt=tbl.new_amount.mean())
    result = compute(expr, s, return_type='native')
    expected = """SELECT
        accounts.name,
        accounts.amount * :amount_1 as one_two,
        avg(accounts.amount + :amount_2) as avg_amt
    FROM
        accounts
    GROUP BY
        accounts.name, accounts.amount * :amount_1
    """
    assert normalize(str(result)) == normalize(expected)


def test_multi_column_by_after_transform_and_filter():
    tbl = t[t.name == 'Alice']
    tbl = transform(tbl, new_amount=tbl.amount + 1, one_two=tbl.amount * 2)
    expr = by(tbl[['name', 'one_two']], avg_amt=tbl.new_amount.mean())
    result = compute(expr, s, return_type='native')
    expected = """SELECT
        accounts.name,
        accounts.amount * :amount_1 as one_two,
        avg(accounts.amount + :amount_2) as avg_amt
    FROM
        accounts
    WHERE
        accounts.name = :name_1
    GROUP BY
        accounts.name, accounts.amount * :amount_1
    """
    assert normalize(str(result)) == normalize(expected)


def test_attribute_access_on_transform_filter():
    tbl = transform(t, new_amount=t.amount + 1)
    expr = tbl[tbl.name == 'Alice'].new_amount
    result = compute(expr, s, return_type='native')
    expected = """SELECT
        accounts.amount + :amount_1 as new_amount
    FROM
        accounts
    WHERE
        accounts.name = :name_1
    """
    assert normalize(str(result)) == normalize(expected)


def test_attribute_on_filter_transform_groupby():
    tbl = t[t.name == 'Alice']
    tbl = transform(tbl, new_amount=tbl.amount + 1, one_two=tbl.amount * 2)
    gb = by(tbl[['name', 'one_two']], avg_amt=tbl.new_amount.mean())
    expr = gb.avg_amt
    result = compute(expr, s, return_type='native')
    expected = """SELECT
        avg(accounts.amount + :amount_1) as avg_amt
    FROM
        accounts
    WHERE
        accounts.name = :name_1
    GROUP BY
        accounts.name, accounts.amount * :amount_2
    """
    assert normalize(str(result)) == normalize(expected)


def test_label_projection():
    tbl = t[(t.name == 'Alice')]
    tbl = transform(tbl, new_amount=tbl.amount + 1, one_two=tbl.amount * 2)
    expr = tbl[['new_amount', 'one_two']]

    # column selection shouldn't affect the resulting SQL
    result = compute(expr[expr.new_amount > 1].one_two, s, return_type='native')
    result2 = compute(expr.one_two[expr.new_amount > 1], s, return_type='native')

    expected = """SELECT
        accounts.amount * :amount_1 as one_two
    FROM accounts
    WHERE accounts.name = :name_1 and accounts.amount + :amount_2 > :param_1
    """
    assert normalize(str(result)) == normalize(expected)
    assert normalize(str(result2)) == normalize(expected)


def test_baseball_nested_by():
    dta = bz_data('sqlite:///%s' % example('teams.db'))
    dshape = discover(dta)
    d = symbol('d', dshape)
    expr = by(d.teams.name,
              start_year=d.teams.yearID.min()).start_year.count_values()
    result = compute(expr, dta, post_compute=False, return_type='native')
    expected = """SELECT
        anon_1.start_year,
        anon_1.count
    FROM
        (SELECT
            alias.start_year as start_year,
            count(alias.start_year) as count
         FROM
            (SELECT
                min(teams.yearid) as start_year
             FROM teams
             GROUP BY teams.name) as alias
         GROUP BY alias.start_year) as anon_1 ORDER BY anon_1.count DESC
    """
    assert normalize(str(result).replace('"', '')) == normalize(expected)


def test_label_on_filter():
    expr = t[t.name == 'Alice'].amount.label('foo').head(2)
    result = compute(expr, s, return_type='native')
    expected = """SELECT
        accounts.amount AS foo
    FROM
        accounts
    WHERE
        accounts.name = :name_1
    LIMIT :param_1
    """
    assert normalize(str(result)) == normalize(expected)


def test_single_field_filter():
    expr = t.amount[t.amount > 0]
    result = compute(expr, s, return_type='native')
    expected = """SELECT
        accounts.amount
    FROM accounts
    WHERE accounts.amount > :amount_1
    """
    assert normalize(str(result)) == normalize(expected)


def test_multiple_field_filter():
    expr = t.name[t.amount > 0]
    result = compute(expr, s, return_type='native')
    expected = """SELECT
        accounts.name
    FROM accounts
    WHERE accounts.amount > :amount_1
    """
    assert normalize(str(result)) == normalize(expected)


def test_distinct_on_label():
    expr = t.name.label('foo').distinct()
    result = compute(expr, s, return_type='native')
    expected = """SELECT
        DISTINCT accounts.name AS foo
    FROM accounts
    """
    assert normalize(str(result)) == normalize(expected)


@pytest.mark.parametrize('n', [-1, 0, 1])
def test_shift_on_column(n):
    expr = t.name.shift(n)
    result = compute(expr, s, return_type='native')
    expected = """SELECT
        lag(accounts.name, :lag_1) over () as name
    FROM accounts
    """
    assert normalize(str(result)) == normalize(expected)


@pytest.fixture
def lhs():
    return pd.DataFrame.from_records(
        dict(zip('cd', row)) for row in zip(
            ['AAPL', 'FB', 'AMZN', 'FB', 'GOOG'] * 2,
            list(np.random.rand(10) * 100)
        )
    )


@pytest.yield_fixture
def rhs():
    try:
        t = bz_data(
            'sqlite:///:memory:::rhs',
            dshape='var * {a: int64, b: ?float64, c: ?string}'
        )
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            t.insert().values([
                dict(zip(t.columns.keys(), row)) for row in zip(
                    range(5),
                    list(np.random.rand(5) * 100),
                    ['AAPL', 'AAPL', 'MSFT', 'FB', 'MSFT']
                )
            ]).execute()
            yield t
        finally:
            drop(t)



@pytest.mark.skipif(LooseVersion(sqlite3.version) <= '2.6.0',
                    reason="Older sqlite3 versions don't support this")
def test_multiple_table_join(lhs, rhs):
    lexpr = symbol('lhs', discover(lhs))
    rexpr = symbol('rhs', discover(rhs))
    expr = join(lexpr, rexpr, 'c')
    result = compute(expr, {lexpr: lhs, rexpr: rhs}, return_type='native')
    expected = pd.merge(lhs, odo(rhs, pd.DataFrame), on='c')
    tm.assert_frame_equal(result, expected)


def test_empty_string_comparison_with_option_type():
    expr = nt.amount[nt.name == '']
    result = compute(expr, s, return_type='native')
    expected = """
    SELECT accounts.amount
    FROM accounts
    WHERE accounts.name = :name_1
    """
    assert normalize(str(result)) == normalize(expected)


def test_tail_no_sort():
    assert (
        normalize(str(compute(t.head(), {t: s}, return_type='native'))) ==
        normalize(str(compute(t.tail(), {t: s}, return_type='native')))
    )


def test_tail_of_sort():
    expected = normalize(str(compute(
        t.sort('id', ascending=False).head(5).sort('id'),
        {t: s}, return_type='native'
    )))
    result = normalize(str(compute(t.sort('id').tail(5), {t: s}, return_type='native')))
    assert expected == result


def test_tail_sort_in_chilren():
    expected = normalize(str(compute(
        t.name.sort('id', ascending=False).head(5).sort('id'),
        {t: s}, return_type='native'
    )))
    result = normalize(str(compute(t.name.sort('id').tail(5), {t: s}, return_type='native')))
    assert expected == result


def test_selection_inner_inputs():
    result = normalize(str(compute(t[t.id == tdate.id], {t: s, tdate: sdate}, return_type='native')))
    expected = normalize("""
    select {a}.name, {a}.amount, {a}.id from {a}, {b} where {a}.id = {b}.id
    """).format(a=s.name, b=sdate.name)
    assert result == expected


@pytest.mark.parametrize('unit', bz_datetime.units)
def test_datetime_trunc(unit):
    ds = 'var * {a: datetime}'
    db = bz_data('sqlite:///:memory:::s', dshape=ds)
    s = symbol('s', ds)

    assert normalize(compute(s.a.truncate(**{unit: 1}), db, return_type='native')) == normalize(
        'select date_trunc(%r, s.a) as a from s' % unit,
    )


@pytest.mark.parametrize('alias,unit', bz_datetime.unit_aliases.items())
def test_datetime_trunc_aliases(alias, unit):
    ds = 'var * {a: datetime}'
    db = bz_data('sqlite:///:memory:::s', dshape=ds)
    s = symbol('s', ds)

    assert normalize(compute(s.a.truncate(**{alias: 1}), db, return_type='native')) == normalize(
        'select date_trunc(%r, s.a) as a from s' % unit,
    )


@pytest.mark.xfail(
    raises=AssertionError,
    reason="We don't currently handle the inner select",
)
def test_inner_select_with_filter():
    ds = 'var * {a: float32}'
    db = bz_data('sqlite:///:memory:::s', dshape=ds)
    s = symbol('s', ds)

    assert normalize(compute(s[s.a == s.a[s.a == 1]].a, db, return_type='native')) == normalize(
        """\
        select
            s.a
        from s, (
            select
                s.a as a
            from
                s
            where
                s.a = 1
        ) as anon_1
        where
            s.a = anon_1.a
        """,
    )


@pytest.mark.parametrize('op', (greatest, least))
def test_greatest(op):
    ds = 'var * {a: int32, b: int32}'
    db = bz_data('sqlite:///:memory:::s', dshape=ds)
    odo([(1, 2)], db)
    s = symbol('s', dshape=ds)

    assert normalize(compute(op(s.a, s.b), db)) == normalize(
        'select {op}(s.a, s.b) as {op}_1 from s'.format(op=op.__name__),
    )


def test_coalesce():
    ds = 'var * {a: ?int32}'
    db = resource('sqlite:///:memory:::s', dshape=ds)
    s = symbol('s', ds)
    t = symbol('s', 'int32')

    assert normalize(compute(coalesce(s.a, t), {s: db, t: 1})) == normalize(
        """\
        select
            coalesce(s.a, 1) as coalesce_1
        from s
        """,
    )
    assert normalize(compute(coalesce(s.a, 1), {s: db})) == normalize(
        """\
        select
            coalesce(s.a, 1) as a
        from s
        """,
    )
