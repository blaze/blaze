import graphlab as gl
import pandas as pd
from odo import odo
from graphlab import aggregate as agg
from blaze import by, compute, Data


def test_by():
    sf = gl.SFrame(
        pd.DataFrame({'a': [1, 2, 3, 4, 5, 6], 'b': list('abcabc')}))
    d = Data(sf)
    expr = by(d.b, a_sum=d.a.sum())
    result = compute(expr)
    expected = sf.groupby('b',
                          operations={'a_sum': agg.SUM('a')})
    assert odo(result, list) == odo(expected, list)
