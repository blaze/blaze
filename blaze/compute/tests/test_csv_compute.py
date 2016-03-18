from blaze.compute.csv import pre_compute, CSV
from blaze import compute, discover, dshape, into, join, concat, data
from blaze.utils import example, filetext, filetexts
from blaze.expr import symbol
from pandas import DataFrame, Series
import pandas.util.testing as tm
from datashape.predicates import iscollection
import numpy as np
import pandas as pd
from toolz import first
from collections import Iterator
from odo import odo
from odo.chunks import chunks


def test_pre_compute_on_small_csv_gives_dataframe():
    csv = CSV(example('iris.csv'))
    s = symbol('s', discover(csv))
    assert isinstance(pre_compute(s.species, csv), (Series, DataFrame))


def test_pre_compute_on_large_csv_gives_chunked_reader():
    csv = CSV(example('iris.csv'))
    s = symbol('s', discover(csv))
    assert isinstance(pre_compute(s.species, csv, comfortable_memory=10),
                      (chunks(pd.DataFrame), pd.io.parsers.TextFileReader))


def test_pre_compute_with_head_on_large_csv_yields_iterator():
    csv = CSV(example('iris.csv'))
    s = symbol('s', discover(csv))
    assert isinstance(pre_compute(s.species.head(), csv, comfortable_memory=10),
                      Iterator)


def test_compute_chunks_on_single_csv():
    csv = CSV(example('iris.csv'))
    s = symbol('s', discover(csv))
    expr = s.sepal_length.max()
    assert compute(expr, {s: csv}, comfortable_memory=10, chunksize=50) == 7.9


def test_pre_compute_with_projection_projects_on_data_frames():
    csv = CSV(example('iris.csv'))
    s = symbol('s', discover(csv))
    result = pre_compute(s[['sepal_length', 'sepal_width']].distinct(),
                         csv, comfortable_memory=10)
    assert set(first(result).columns) == \
            set(['sepal_length', 'sepal_width'])


def test_pre_compute_calls_lean_projection():
    csv = CSV(example('iris.csv'))
    s = symbol('s', discover(csv))
    result = pre_compute(s.sort('sepal_length').species,
                         csv, comfortable_memory=10)
    assert set(first(result).columns) == \
            set(['sepal_length', 'species'])


def test_unused_datetime_columns():
    ds = dshape('2 * {val: string, when: datetime}')
    with filetext("val,when\na,2000-01-01\nb,2000-02-02") as fn:
        csv = CSV(fn, has_header=True)

        s = symbol('s', discover(csv))
        assert into(list, compute(s.val, csv)) == ['a', 'b']


def test_multiple_csv_files():
    d = {'mult1.csv': 'name,val\nAlice,1\nBob,2',
         'mult2.csv': 'name,val\nAlice,3\nCharlie,4'}

    dta = [('Alice', 1), ('Bob', 2), ('Alice', 3), ('Charlie', 4)]
    with filetexts(d) as fns:
        r = data('mult*.csv')
        s = symbol('s', discover(r))

        for e in [s, s.name, s.name.nunique(), s.name.count_values(),
                s.val.mean()]:
            a = compute(e, {s: r})
            b = compute(e, {s: dta})
            if iscollection(e.dshape):
                a, b = into(set, a), into(set, b)
            assert a == b


def test_csv_join():
    d = {'a.csv': 'a,b,c\n0,1,2\n3,4,5',
         'b.csv': 'c,d,e\n2,3,4\n5,6,7'}

    with filetexts(d):
        data_a = data('a.csv')
        data_b = data('b.csv')
        a = symbol('a', discover(data_a))
        b = symbol('b', discover(data_b))
        tm.assert_frame_equal(
            odo(
                compute(join(a, b, 'c'), {a: data_a, b: data_b}),
                pd.DataFrame,
            ),

            # windows needs explicit int64 construction b/c default is int32
            pd.DataFrame(np.array([[2, 0, 1, 3, 4],
                                   [5, 3, 4, 6, 7]], dtype='int64'),
                         columns=list('cabde'))
        )


def test_concat():
    d = {'a.csv': 'a,b\n1,2\n3,4',
         'b.csv': 'a,b\n5,6\n7,8'}

    with filetexts(d):
        a_rsc = data('a.csv')
        b_rsc = data('b.csv')

        a = symbol('a', discover(a_rsc))
        b = symbol('b', discover(b_rsc))

        tm.assert_frame_equal(
            odo(
                compute(concat(a, b), {a: a_rsc, b: b_rsc}), pd.DataFrame,
            ),

            # windows needs explicit int64 construction b/c default is int32
            pd.DataFrame(np.arange(1, 9, dtype='int64').reshape(4, 2),
                         columns=list('ab')),
        )
