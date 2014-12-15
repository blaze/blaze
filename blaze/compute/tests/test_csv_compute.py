from blaze.compute.csv import pre_compute, CSV
from blaze import compute, discover, dshape, into
from blaze.utils import example, filetext
from blaze.expr import Expr, symbol
from pandas import DataFrame, Series
import pandas as pd
from toolz import first
from collections import Iterator
from into.chunks import chunks

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
