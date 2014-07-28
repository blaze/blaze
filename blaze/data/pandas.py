from __future__ import absolute_import, division, print_function

import pandas as pd
from pandas import DataFrame
from ..dispatch import dispatch
from .csv import CSV
from toolz import valmap
from datashape import to_numpy_dtype


@dispatch(DataFrame, CSV)
def into(a, b):
    dialect= b.dialect.copy()
    del dialect['lineterminator']
    dates = [i for i, typ in enumerate(b.schema[0].types)
               if 'date' in str(typ)]
    dtypes = valmap(to_numpy_dtype, b.schema[0].dict)
    return pd.read_csv(b.path,
                       names=b.columns,
                       parse_dates=dates,
                       dtype=dtypes,
                       **dialect)
