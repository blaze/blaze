import pytest

import os

from blaze import data, compute
from blaze.utils import raises
from odo import URL, CSV

import pandas as pd
import pandas.util.testing as tm

from functools import partial

try:
    from urllib2 import urlopen
    from urllib2 import HTTPError, URLError
except ImportError:
    from urllib.request import urlopen
    from urllib.error import HTTPError, URLError

pytestmark = pytest.mark.skipif(raises(URLError,
                                       partial(urlopen, "http://google.com")),
                                reason='unable to connect to google.com')

iris_url = ('https://raw.githubusercontent.com/'
            'blaze/blaze/master/blaze/examples/data/iris.csv')

@pytest.fixture
def iris_local():
    thisdir = os.path.abspath(os.path.dirname(__file__))
    return data(os.path.join(thisdir, os.pardir, os.pardir, "examples", "data", "iris.csv"))

def test_url_csv_data(iris_local):
    iris_remote = data(iris_url)
    assert isinstance(iris_remote.data, URL(CSV))
    iris_remote_df = compute(iris_remote)
    assert isinstance(iris_remote_df, pd.DataFrame)
    iris_local_df = compute(iris_local)
    tm.assert_frame_equal(iris_remote_df, iris_local_df)
