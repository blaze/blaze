from blaze.compute.csv import pre_compute, CSV
from blaze import compute
from blaze.utils import example
from blaze.expr import Expr, symbol
from pandas import DataFrame
import pandas

def test_pre_compute_on_small_csv_gives_dataframe():
    csv = CSV(example('iris.csv'))
    s = symbol('s', csv.dshape)
    assert isinstance(pre_compute(s.species, csv), DataFrame)


def test_pre_compute_on_large_csv_gives_chunked_reader():
    csv = CSV(example('iris.csv'))
    s = symbol('s', csv.dshape)
    assert isinstance(pre_compute(s.species, csv, comfortable_memory=10),
                      pandas.io.parsers.TextFileReader)


def test_compute_chunks_on_single_csv():
    csv = CSV(example('iris.csv'))
    s = symbol('s', csv.dshape)
    expr = s.sepal_length.max()
    assert compute(expr, {s: csv}, comfortable_memory=10, chunksize=50) == 7.9
