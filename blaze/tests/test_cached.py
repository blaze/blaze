from blaze.cached import CachedDataset
from blaze import symbol, discover, compute, into
import pandas as pd
from collections import Iterator


df = pd.DataFrame([['Alice', 100, 1],
                   ['Bob', 200, 2],
                   ['Alice', 50, 3]],
                  columns=['name', 'amount', 'id'])


t = symbol('t', discover(df))


def test_dataset():
    ns = {'t': df, 'x': 10}
    cache=dict()
    d = CachedDataset(ns, cache=cache)

    assert discover(d) == discover(ns)

    s = symbol('s', discover(d))
    compute(s.x * 2, d) == 20
    cache == {s.x * 2: 20}


def test_streaming():
    seq = [{'name': 'Alice', 'x': 1},
           {'name': 'Bob', 'x': 1}]
    ns = {'t': seq, 'x': 10}
    cache=dict()
    d = CachedDataset(ns, cache=cache)

    s = symbol('s', discover(d))
    expr = s.t.x * 2
    result = compute(expr, d)

    assert not isinstance(d.cache[expr], Iterator)
    assert into(list, d.cache[expr]) == [2, 2]
