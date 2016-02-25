from blaze import compute, data, symbol, discover
from blaze.utils import example


flag = [False]

def mymap(func, *args):
    flag[0] = True
    return map(func, *args)


def test_map_called_on_data_star():
    r = data(example('accounts_*.csv'))
    s = symbol('s', discover(r))
    flag[0] = False
    a = compute(s.count(), r)
    b = compute(s.count(), r, map=mymap)
    assert a == b
    assert flag[0]
