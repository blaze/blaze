from __future__ import absolute_import, division, print_function

from blaze.api.into import *
from blaze.data.pandas import *
from pandas import DataFrame
from blaze.utils import filetext
import numpy as np

def test_into():
    with filetext('1,2\n3,4\n') as fn:
        csv = CSV(fn, schema='{a: int64, b: float64}')
        df = into(DataFrame, csv)

        expected = DataFrame([[1, 2.0], [3, 4.0]],
                             columns=['a', 'b'])


        assert str(df) == str(expected)
        print(df.dtypes)
        assert list(df.dtypes) == [np.int64, np.float64]

