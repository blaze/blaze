from __future__ import absolute_import, division, print_function

from blaze.compute.spark import *
from blaze.compatibility import skip
from blaze.expr.table import *

data = [['Alice', 100, 1],
        ['Bob', 200, 2],
        ['Alice', 50, 3]]
data2 = [['Alice', 'Austin'],
         ['Bob', 'Boston']]
try:
    from pyspark import SparkContext
    sc = SparkContext("local", "Simple App")
    rdd = sc.parallelize(data)
    rdd2 = sc.parallelize(data2)
except ImportError:
    pass

t = TableSymbol('t', '{name: string, amount: int, id: int}')

t2 = TableSymbol('t2', '{name: string, city: string}')

#Web Commons Graph Example data
data_idx = [['A', 1],
            ['B', 2],
            ['C', 3]]

data_arc = [[1, 3],
            [2, 3],
            [3, 1]]

t_idx = TableSymbol('idx', '{name: string, node_id: int32}')

t_arc = TableSymbol('arc', '{node_out: int32, node_id: int32}')

def test_table():
    assert compute(t, rdd) == rdd


def test_projection():
    assert compute(t['name'], rdd).collect() == [row[0] for row in data]


def test_multicols_projection():
    result = compute(t[['amount', 'name']], rdd).collect()
    expected = [(100, 'Alice'), (200, 'Bob'), (50, 'Alice')]

    print(result)
    print(expected)

    assert result == expected


inc = lambda x: x + 1


reduction_exprs = [
    t['amount'].sum(),
    t['amount'].min(),
    t['amount'].max(),
    t['amount'].nunique(),
    t['name'].nunique(),
    t['amount'].count(),
    (t['amount'] > 150).any(),
    (t['amount'] > 150).all(),
    t['amount'].mean(),
    t['amount'].var(),
    t['amount'].std()]


def test_reductions():
    for expr in reduction_exprs:
        result = compute(expr, rdd)
        expected = compute(expr, data)
        if not result == expected:
            print(result)
            print(expected)
            if isinstance(result, float):
                assert abs(result - expected) < 0.001
            else:
                assert result == expected

exprs = [
    t['amount'],
    t['amount'] == 100,
    t[t['name'] == 'Alice'],
    t[t['amount'] == 0],
    t[t['amount'] > 150],
    t['amount'] + t['id'],
    t['amount'] % t['id'],
    exp(t['amount']),
    By(t, t['name'], t['amount'].sum()),
    By(t, t['name'], (t['amount'] + 1).sum()),
    (t['amount'] * 1).label('foo'),
    t.map(lambda _, amt, id: amt + id),
    t['amount'].map(inc)]

"""
big_exprs = [
    By(tbig, tbig[['name', 'sex']], tbig['amount'].sum()),
    By(tbig, tbig[['name', 'sex']], (tbig['id'] + tbig['amount']).sum())]
"""


def test_basic():
    check_exprs_against_python(exprs)


def check_exprs_against_python(exprs):
    any_bad = False
    for expr in exprs:
        result = compute(expr, rdd).collect()
        expected = list(compute(expr, data))
        if not result == expected:
            any_bad = True
            print("Expression:", expr)
            print("Spark:", result)
            print("Python:", expected)

    assert not any_bad


def test_head():
    assert list(compute(t.head(1), rdd)) == list(compute(t.head(1), data))


def test_sort():
    check_exprs_against_python([
                t.sort('amount'),
                t.sort('amount', ascending=True),
                t.sort(['amount', 'id'])])


def test_join():

    joined = Join(t, t2, 'name')
    expected = [['Alice', 100, 1, 'Austin'],
                ['Bob', 200, 2, 'Boston'],
                ['Alice', 50, 3, 'Austin']]
    result = compute(joined, rdd, rdd2).collect()
    assert all(i in expected for i in result)


def test_groupby():
    rddidx = sc.parallelize(data_idx)
    rddarc = sc.parallelize(data_arc)

    joined = Join(t_arc, t_idx, "node_id")

    result_blaze = compute(joined, {t_arc: rddarc, t_idx:rddidx})
    t = By(joined, joined['name'], joined['node_id'].count())
    a = compute(t, {t_arc: rddarc, t_idx:rddidx})
    in_degree = dict(a.collect())
    assert in_degree == {'A': 1, 'C': 2}


@skip("Spark not yet fully supported")
def test_jaccard():
    data_idx_j = sc.parallelize([['A', 1],['B', 2],['C', 3],['D', 4],['E', 5],['F', 6]])
    data_arc_j = sc.parallelize([[1, 3],[2, 3],[4, 3],[5, 3],[3, 1],[2, 1],[5, 1],[1, 6],[2, 6],[4, 6]])

    #The tables we need to work with
    t_idx_j = TableSymbol('{name: string, node_id: int32}') #Index of sites
    t_arc_j = TableSymbol('{node_out: int32, node_id: int32}') # Links between sites
    t_sel_j = TableSymbol('{name: string}') # A Selection table for just site names

    join_names = Join(t_arc_j, t_idx_j, "node_id")
    user_selected = Join(join_names, t_sel_j, "name")
    proj_of_nodes = user_selected[['node_out', 'node_id']]
    node_selfjoin = Join(proj_of_nodes, proj_of_nodes.relabel(
        {'node_id':'node_other'}), "node_out")
    #Filter here to get (a,b) node pairs where a < b
    flter = node_selfjoin[ node_selfjoin['node_id'] < node_selfjoin['node_other']]
    gby = By(flter, flter[['node_id', 'node_other']], flter['node_out'].count())
    indeg_joined = Join(t_arc, t_idx, 'node_id')
    indeg_t = By(indeg_joined, indeg_joined['node_id'], indeg_joined['node_id'].count())

    #### Now we actually do the computation on the graph:
    # The subset we care about
    data_sel_j = sc.parallelize([['C'],['F']])
    shared_neighbor_num = compute(gby, {t_sel_j: data_sel_j, t_arc:data_arc_j, t_idx_j:data_idx_j})
    indeg = compute(indeg_t, {t_arc_j: data_arc_j, t_idx_j:data_idx_j})
    indeg_py = dict(indeg.collect())
    shared_neighbor_py = shared_neighbor_num.collect()
    assert shared_neighbor_py == [((3, 6), 3)]
    assert indeg_py == {1: 3, 3: 4, 6: 3}
