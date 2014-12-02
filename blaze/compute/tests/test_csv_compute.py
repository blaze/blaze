from blaze.compute.csv import pre_compute, CSV
from blaze import compute
from blaze.utils import example
from blaze.expr import Expr, symbol
from pandas import DataFrame
import pandas
from collections import Iterator

def test_pre_compute_on_small_csv_gives_dataframe():
    csv = CSV(example('iris.csv'))
    s = symbol('s', csv.dshape)
    assert isinstance(pre_compute(s.species, csv), DataFrame)


def test_pre_compute_on_large_csv_gives_chunked_reader():
    csv = CSV(example('iris.csv'))
    s = symbol('s', csv.dshape)
    assert isinstance(pre_compute(s.species, csv, comfortable_memory=10),
                      pandas.io.parsers.TextFileReader)


def test_pre_compute_with_head_on_large_csv_yields_iterator():
    csv = CSV(example('iris.csv'))
    s = symbol('s', csv.dshape)
    assert isinstance(pre_compute(s.species.head(), csv, comfortable_memory=10),
                      Iterator)


def test_compute_chunks_on_single_csv():
    csv = CSV(example('iris.csv'))
    s = symbol('s', csv.dshape)
    expr = s.sepal_length.max()
    assert compute(expr, {s: csv}, comfortable_memory=10, chunksize=50) == 7.9


def test_pre_compute_with_projection_projects_on_data_frames():
    csv = CSV(example('iris.csv'))
    s = symbol('s', csv.dshape)
    result = pre_compute(s[['sepal_length', 'sepal_width']].distinct(),
                         csv, comfortable_memory=10)
    assert set(result.get_chunk().columns) == \
            set(['sepal_length', 'sepal_width'])


def test_pre_compute_calls_lean_projection():
    csv = CSV(example('iris.csv'))
    s = symbol('s', csv.dshape)
    result = pre_compute(s.sort('sepal_length').species,
                         csv, comfortable_memory=10)
    assert set(result.get_chunk().columns) == \
            set(['sepal_length', 'species'])
