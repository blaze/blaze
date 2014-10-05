from blaze.expr import *
from blaze.expr.split import *
from blaze.api.dplyr import transform
import datashape
from datashape.predicates import isscalar

t = TableSymbol('t', '{name: string, amount: int, id: int}')


def test_path_split():
    expr = t.amount.sum() + 1
    assert path_split(t, expr).isidentical(t.amount.sum())

    expr = t.amount.distinct().sort()
    assert path_split(t, expr).isidentical(t.amount.distinct())

    t2 = transform(t, id=t.id * 2)
    expr = by(t2.id, amount=t2.amount.sum()).amount + 1
    assert path_split(t, expr).isidentical(by(t2.id, amount=t2.amount.sum()))

    expr = count(t.amount.distinct())
    assert path_split(t, expr).isidentical(t.amount.distinct())

    expr = summary(total=t.amount.sum())
    assert path_split(t, expr).isidentical(expr)


def test_sum():
    (chunk, chunk_expr), (agg, agg_expr) = split(t, t.amount.sum())

    assert chunk.schema == t.schema
    assert chunk_expr.isidentical(chunk.amount.sum())

    assert isscalar(agg.dshape.measure)
    assert agg_expr.isidentical(sum(agg))


def test_distinct():
    (chunk, chunk_expr), (agg, agg_expr) = split(t, count(t.amount.distinct()))

    assert chunk.schema == t.schema
    assert chunk_expr.isidentical(chunk.amount.distinct())

    assert isscalar(agg.dshape.measure)
    assert agg_expr.isidentical(count(agg.distinct()))


def test_summary():
    (chunk, chunk_expr), (agg, agg_expr) = split(t, summary(a=t.amount.count(),
                                                            b=t.id.sum() + 1))

    assert chunk.schema == t.schema
    assert chunk_expr.isidentical(summary(a=chunk.amount.count(),
                                          b=chunk.id.sum()))

    assert not agg.schema == dshape('{a: int32, b: int32}')
    assert agg_expr.isidentical(summary(a=agg.a.sum(),
                                        b=agg.b.sum() + 1))

    (chunk, chunk_expr), (agg, agg_expr) = \
            split(t, summary(total=t.amount.sum()))

    assert chunk_expr.isidentical(summary(total=chunk.amount.sum()))
    assert agg_expr.isidentical(summary(total=agg.total.sum()))


def test_by_sum():
    (chunk, chunk_expr), (agg, agg_expr) = \
            split(t, by(t.name, total=t.amount.sum()))

    assert chunk.schema == t.schema
    assert chunk_expr.isidentical(by(chunk.name, total=chunk.amount.sum()))

    assert not isscalar(agg.dshape.measure)
    assert agg_expr.isidentical(by(agg.name, total=agg.total.sum()))

def test_by_count():
    (chunk, chunk_expr), (agg, agg_expr) = \
            split(t, by(t.name, total=t.amount.count()))

    assert chunk_expr.isidentical(by(chunk.name, total=chunk.amount.count()))

    assert agg_expr.isidentical(by(agg.name, total=agg.total.sum()))


def test_embarassing_rowwise():
    (chunk, chunk_expr), (agg, agg_expr) = split(t, t.amount + 1)

    assert chunk_expr.isidentical(chunk.amount + 1)
    assert agg_expr.isidentical(agg)


def test_embarassing_selection():
    (chunk, chunk_expr), (agg, agg_expr) = split(t, t[t.amount > 0])

    assert chunk_expr.isidentical(chunk[chunk.amount > 0])
    assert agg_expr.isidentical(agg)
