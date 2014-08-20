from blaze.api.dplyr import *

L = [[1, 'Alice',   100],
     [2, 'Bob',    -200],
     [3, 'Charlie', 300],
     [4, 'Dennis',  400],
     [5, 'Edith',  -500]]

t = TableSymbol('t', '{id: int, name: string, amount: int}')

inject(t)

def test_Filter():
    assert Filter(t, name=='Alice').isidentical(t[t.name=='Alice'])
    assert Filter(t, name=='Alice', amount>1).isidentical(
            t[(t.name=='Alice') & (t.amount > 1)])


def test_select():
    assert select(t, name, amount).isidentical(t[['name', 'amount']])


def test_mutate():
    expr = mutate(t, x = amount + id)
    assert expr.columns == ['id', 'name', 'amount', 'x']
#     assert expr.x.isidentical(t.amount + t.id)


def test_groupby():
    g = group_by(t, name)
    result = summarize(g, sum=amount.sum())
    assert isinstance(result, By)
    assert result.columns == ['name', 'sum']


def test_groupby_multicolumn():
    g = group_by(t, name, id)
    result = summarize(g, count=amount.count())
    assert isinstance(result, By)
    assert result.columns == ['name', 'id', 'count']
