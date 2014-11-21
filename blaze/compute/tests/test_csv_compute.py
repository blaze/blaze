from blaze.compute.csv import pre_compute, CSV
from blaze.utils import example
from blaze.expr import Expr, symbol
from pandas import DataFrame

def test_pre_compute_on_small_csv_gives_dataframe():
    csv = CSV(example('iris.csv'))
    s = symbol('s', csv.dshape)
    assert isinstance(pre_compute(s.species, csv), DataFrame)
