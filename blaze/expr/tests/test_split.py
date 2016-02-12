import pytest
from blaze.expr import (symbol, transform, by, count, summary, var, std, mean,
                        sqrt, sum)
from blaze.expr.split import split, path_split
from datashape import dshape
from datashape.predicates import isscalar, isrecord, iscollection

t = symbol('t', 'var * {name: string, amount: int32, id: int32}')
a = symbol('a', '1000 * 2000 * {x: float32, y: float32}')


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
    assert chunk_expr.isidentical(chunk.amount.sum(keepdims=True))

    assert isscalar(agg.dshape.measure)
    assert agg_expr.isidentical(sum(agg))


def test_mean():
    (chunk, chunk_expr), (agg, agg_expr) = split(t, t.amount.mean())

    assert chunk.schema == t.schema
    assert chunk_expr.isidentical(summary(total=chunk.amount.sum(),
                                          count=chunk.amount.count(),
                                          keepdims=True))

    assert isrecord(agg.dshape.measure)
    assert agg_expr.isidentical(agg.total.sum() / agg['count'].sum())


def test_var():
    (chunk, chunk_expr), (agg, agg_expr) = split(t, t.amount.var())

    assert chunk.schema == t.schema
    assert chunk_expr.isidentical(summary(x=chunk.amount.sum(),
                                          x2=(chunk.amount ** 2).sum(),
                                          n=chunk.amount.count(),
                                          keepdims=True))

    assert isrecord(agg.dshape.measure)
    assert agg_expr.isidentical((agg.x2.sum() / (agg.n.sum())
                                 - (agg.x.sum() / (agg.n.sum())) ** 2))


def test_std():
    (chunk, chunk_expr), (agg, agg_expr) = split(t, t.amount.std())

    assert chunk.schema == t.schema
    assert chunk_expr.isidentical(summary(x=chunk.amount.sum(),
                                          x2=(chunk.amount ** 2).sum(),
                                          n=chunk.amount.count(),
                                          keepdims=True))

    assert isrecord(agg.dshape.measure)
    assert agg_expr.isidentical(sqrt((agg.x2.sum() / (agg.n.sum())
                                      - (agg.x.sum() / (agg.n.sum())) ** 2)))


def test_sum_with_axis_argument():
    chunk = symbol('chunk', '100 * 100 * {x: float32, y: float32}')
    (chunk, chunk_expr), (agg, agg_expr) = split(
        a, a.x.sum(axis=0), chunk=chunk)

    assert chunk.schema == a.schema
    assert agg_expr.dshape == a.x.sum(axis=0).dshape

    assert chunk_expr.isidentical(chunk.x.sum(axis=0, keepdims=True))
    assert agg_expr.isidentical(agg.sum(axis=0))


def test_sum_with_keepdims():
    (chunk, chunk_expr), (agg, agg_expr) = split(
        t, t.amount.sum(keepdims=True))

    assert chunk.schema == t.schema
    assert chunk_expr.isidentical(chunk.amount.sum(keepdims=True))

    assert isscalar(agg.dshape.measure)
    assert agg_expr.isidentical(sum(agg, keepdims=True))


def test_split_reasons_correctly_about_uneven_aggregate_shape():
    x = symbol('chunk', '10 * 10 * int')
    chunk = symbol('chunk', '3 * 3 * int')
    (chunk, chunk_expr), (agg, agg_expr) = split(x, x.sum(axis=0),
                                                 chunk=chunk)
    assert agg.shape == (4, 10)

    x = symbol('leaf', '1643 * 60 * int')
    chunk = symbol('chunk', '40 * 60 * int')
    (chunk, chunk_expr), (agg, agg_expr) = split(x, x.sum(),
                                                 chunk=chunk)
    assert agg.shape == (42, 1)


def test_split_reasons_correctly_about_aggregate_shape():
    chunk = symbol('chunk', '100 * 100 * {x: float32, y: float32}')
    (chunk, chunk_expr), (agg, agg_expr) = split(a, a.x.sum(), chunk=chunk)

    assert agg.shape == (10, 20)

    chunk = symbol('chunk', '100 * 100 * {x: float32, y: float32}')
    (chunk, chunk_expr), (agg, agg_expr) = split(
        a, a.x.sum(axis=0), chunk=chunk)

    assert agg.shape == (10, 2000)


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
                                          b=chunk.id.sum(), keepdims=True))

    # assert not agg.schema == dshape('{a: int32, b: int32}')
    assert agg_expr.isidentical(summary(a=agg.a.sum(),
                                        b=agg.b.sum() + 1))

    (chunk, chunk_expr), (agg, agg_expr) = \
        split(t, summary(total=t.amount.sum()))

    assert chunk_expr.isidentical(summary(total=chunk.amount.sum(),
                                          keepdims=True))
    assert agg_expr.isidentical(summary(total=agg.total.sum()))


def test_summary_with_mean():
    (chunk, chunk_expr), (agg, agg_expr) = split(t, summary(a=t.amount.count(),
                                                            b=t.id.mean() + 1))

    assert chunk.schema == t.schema
    assert chunk_expr.isidentical(summary(a=chunk.amount.count(),
                                          b_total=chunk.id.sum(),
                                          b_count=chunk.id.count(),
                                          keepdims=True))

    # assert not agg.schema == dshape('{a: int32, b: int32}')
    expected = summary(a=agg.a.sum(),
                       b=(agg.b_total.sum() / agg.b_count.sum()) + 1)
    assert agg_expr.isidentical(expected)


def test_complex_summaries():
    t = symbol('t', '100 * {a: int, b: int}')
    (chunk, chunk_expr), (agg, agg_expr) = split(t, summary(q=t.a.mean(),
                                                            w=t.a.std(),
                                                            e=t.a.sum()))

    assert chunk_expr.isidentical(summary(e=chunk.a.sum(),
                                          q_count=chunk.a.count(),
                                          q_total=chunk.a.sum(),
                                          w_n=chunk.a.count(),
                                          w_x=chunk.a.sum(),
                                          w_x2=(chunk.a ** 2).sum(),
                                          keepdims=True))

    expected = summary(e=agg.e.sum(),
                       q=agg.q_total.sum() / agg.q_count.sum(),
                       w=sqrt((agg.w_x2.sum() / agg.w_n.sum())
                              - (agg.w_x.sum() / agg.w_n.sum()) ** 2))
    assert agg_expr.isidentical(expected)


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


def test_by_mean():
    (chunk, chunk_expr), (agg, agg_expr) = \
        split(t, by(t.name, avg=t.amount.mean()))

    assert chunk_expr.isidentical(by(chunk.name,
                                     avg_total=chunk.amount.sum(),
                                     avg_count=chunk.amount.count()))

    assert agg_expr.isidentical(by(agg.name,
                                   avg=(agg.avg_total.sum() / agg.avg_count.sum())))


def test_embarassing_rowwise():
    (chunk, chunk_expr), (agg, agg_expr) = split(t, t.amount + 1)

    assert chunk_expr.isidentical(chunk.amount + 1)
    assert agg_expr.isidentical(agg)


def test_embarassing_selection():
    (chunk, chunk_expr), (agg, agg_expr) = split(t, t[t.amount > 0])

    assert chunk_expr.isidentical(chunk[chunk.amount > 0])
    assert agg_expr.isidentical(agg)


def test_embarassing_like():
    (chunk, chunk_expr), (agg, agg_expr) = split(t, t[t.name.like('Alice*')])

    assert chunk_expr.isidentical(chunk[chunk.name.like('Alice*')])
    assert agg_expr.isidentical(agg)


x = symbol('x', '24 * 16 * int32')


def test_nd_chunk():
    c = symbol('c', '4 * 4 * int32')

    (chunk, chunk_expr), (agg, agg_expr) = split(x, x.sum(), chunk=c)

    assert chunk.shape == (4, 4)
    assert chunk_expr.isidentical(chunk.sum(keepdims=True))

    assert agg.shape == (6, 4)
    assert agg_expr.isidentical(agg.sum())


def test_nd_chunk_axis_args():
    c = symbol('c', '4 * 4 * int32')

    (chunk, chunk_expr), (agg, agg_expr) = split(x, x.sum(axis=0), chunk=c)

    assert chunk.shape == (4, 4)
    assert chunk_expr.shape == (1, 4)
    assert chunk_expr.isidentical(chunk.sum(keepdims=True, axis=0))

    assert agg.shape == (6, 16)
    assert agg_expr.isidentical(agg.sum(axis=0))

    for func in [var, std, mean]:
        (chunk, chunk_expr), (agg, agg_expr) = split(
            x, func(x, axis=0), chunk=c)

        assert chunk.shape == (4, 4)
        assert chunk_expr.shape == (1, 4)
        assert agg.shape == (6, 16)


def test_agg_shape_in_tabular_case_with_explicit_chunk():
    t = symbol('t', '1000 * {name: string, amount: int, id: int}')
    c = symbol('chunk', 100 * t.schema)

    expr = by(t.name, total=t.amount.sum())
    (chunk, chunk_expr), (agg, agg_expr) = split(t, expr, chunk=c)

    assert agg.dshape == dshape('var * {name: string, total: int64}')


def test_reductions():
    (chunk, chunk_expr), (agg, agg_expr) = split(t, t.amount.nunique())

    assert chunk.schema == t.schema
    assert chunk_expr.isidentical(chunk.amount.distinct())

    assert isscalar(agg.dshape.measure)
    assert agg_expr.isidentical(agg.distinct().count())

    (chunk, chunk_expr), (agg, agg_expr) = \
        split(t, t.amount.nunique(keepdims=True))

    assert chunk.schema == t.schema
    assert chunk_expr.isidentical(chunk.amount.distinct())

    assert isscalar(agg.dshape.measure)
    assert agg_expr.isidentical(agg.distinct().count(keepdims=True))


def test_by_with_single_field_child():
    x = symbol('x', 'var * int')
    (chunk, chunk_expr), (agg, agg_expr) = split(x, by(x, total=x.sum()))

    assert chunk_expr.isidentical(by(chunk, total=chunk.sum()))

    assert (agg_expr.isidentical(by(agg[agg.fields[0]],
                                    total=agg.total.sum())
            .relabel({agg.fields[0]: 'x'})))


def test_keepdims_equals_true_doesnt_mess_up_agg_shape():
    x = symbol('x', '10 * int')
    (chunk, chunk_expr), (agg, agg_expr) = split(x, x.sum(), keepdims=False)

    assert iscollection(agg.dshape)


def test_splittable_apply():
    def f(x):
        pass

    (chunk, chunk_expr), (agg, agg_expr) = \
        split(t, t.amount.apply(f, 'var * int', splittable=True))
    assert chunk_expr.isidentical(
        chunk.amount.apply(f, 'var * int', splittable=True))

    assert agg_expr.isidentical(agg)


@pytest.mark.xfail(raises=AssertionError,
                   reason="Can't split on expressions requiring branching")
def test_elemwise_with_multiple_paths():
    s = symbol('s', 'var * {x: int, y: int, z: int}')
    expr = s.x.sum() / s.y.sum()

    (chunk, chunk_expr), (agg, agg_expr) = split(s, expr)
    assert chunk_expr.isidentical(summary(x=chunk.x.sum(), y=chunk.y.sum()))
    assert agg_expr.isidentical(agg.x / agg.y)
