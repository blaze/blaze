import unittest

from dynd import nd
import numpy as np
import pandas as pd
from datashape import dshape
from datetime import datetime
import tempfile
import os

from blaze.api.csv_into import csv_into
import blaze
from blaze import Table
import pytest


def skip(test_foo):
    return


def skip_if_not(x):
    def maybe_a_test_function(test_foo):
        if not x:
            return
        else:
            return test_foo
    return maybe_a_test_function

try:
    from pandas import DataFrame
except ImportError:
    DataFrame = None

try:
    from blaze.data import CSV
except ImportError:
    CSV = None

try:
    from bokeh.objects import ColumnDataSource
except ImportError:
    ColumnDataSource = None


@pytest.yield_fixture
def bad_csv():

    with tempfile.NamedTemporaryFile(mode='w') as f:
        badfile = open(f.name, mode="w")
        # Insert a new record
        badfile.write("a,b,c,d\n")
        badfile.write("e,f,g,h\n")
        badfile.write("i,j,k,l,m\n")
        badfile.flush()
        yield badfile
        # Close (and flush) the file
        badfile.close()


@skip_if_not(CSV and DataFrame)
def test_csv_into_dataframe(bad_csv):
    df = csv_into(pd.DataFrame(), bad_csv.name)
    assert len(df) == 2


@skip_if_not(CSV and DataFrame)
def test_csv_into_dataframe_errors(bad_csv):
    df, err = csv_into(pd.DataFrame(), bad_csv.name, permit_errors=True)
    assert len(df) == 2
    assert err
