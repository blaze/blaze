from blaze import compute, resource, Symbol, discover
from blaze.utils import example


flag = [False]

def mymap(func, *args):
    flag[0] = True
    return map(func, *args)


def test_map_called_on_resource_star():
    r = resource(example('accounts*.csv'))
    s = Symbol('s', discover(r))
    flag[0] = False
    assert compute(s.count(), r) == compute(s.count(), r, map=mymap)
    assert flag[0]
