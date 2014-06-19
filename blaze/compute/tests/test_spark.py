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

def test_spark_table():
    assert compute(t, rdd) == rdd


def test_spark_projection():
    assert compute(t['name'], rdd).collect() == [row[0] for row in data]


def test_spark_multicols_projection():
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


def test_spark_reductions():
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


def test_spark_basic():
    check_exprs_against_python(exprs, data, rdd)


def check_exprs_against_python(exprs, data, rdd):
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


def test_spark_big_by():
    tbig = TableSymbol('tbig', '{name: string, sex: string[1], amount: int, id: int}')

    big_exprs = [
        By(tbig, tbig[['name', 'sex']], tbig['amount'].sum()),
        By(tbig, tbig[['name', 'sex']], (tbig['id'] + tbig['amount']).sum())]

    databig = [['Alice', 'F', 100, 1],
               ['Alice', 'F', 100, 3],
               ['Drew', 'F', 100, 4],
               ['Drew', 'M', 100, 5],
               ['Drew', 'M', 200, 5]]

    rddbig = sc.parallelize(databig)

    check_exprs_against_python(big_exprs, databig, rddbig)


def test_spark_head():
    assert list(compute(t.head(1), rdd)) == list(compute(t.head(1), data))


def test_spark_sort():
    check_exprs_against_python([
                t.sort('amount'),
                t.sort('amount', ascending=True),
                t.sort(['amount', 'id'])], data, rdd)

def test_spark_distinct():
    assert set(compute(t['name'].distinct(), rdd).collect()) == \
            set(['Alice', 'Bob'])



def test_spark_join():

    joined = Join(t, t2, 'name')
    expected = [['Alice', 100, 1, 'Austin'],
                ['Bob', 200, 2, 'Boston'],
                ['Alice', 50, 3, 'Austin']]
    result = compute(joined, rdd, rdd2).collect()
    assert all(i in expected for i in result)


def test_spark_groupby():
    rddidx = sc.parallelize(data_idx)
    rddarc = sc.parallelize(data_arc)

    joined = Join(t_arc, t_idx, "node_id")

    result_blaze = compute(joined, {t_arc: rddarc, t_idx:rddidx})
    t = By(joined, joined['name'], joined['node_id'].count())
    a = compute(t, {t_arc: rddarc, t_idx:rddidx})
    in_degree = dict(a.collect())
    assert in_degree == {'A': 1, 'C': 2}


def test_spark_multi_level_rowfunc_works():
    expr = t['amount'].map(lambda x: x + 1)

    assert compute(expr, rdd).collect() == [x[1] + 1 for x in data]


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


def test_spark_merge():
    col = (t['amount'] * 2).label('new')
    expr = merge(t['name'], col)

    assert compute(expr, rdd).collect() == [(row[0], row[1] * 2) for row in data]


def test_spark_into():
    from blaze.api.into import into
    seq = [1, 2, 3]
    assert isinstance(into(rdd, seq), RDD)
    assert into([], into(rdd, seq)) == seq

def test_spark_select_filter():
    # TODO: work-around issue with filter/select for now
    #t2 = t['name'][t['name'] != 'Alice']
    t2 = t[t['name'] != 'Alice']['name']
    ansrdd = compute(t2, rdd)
    assert ansrdd.collect() == ['Bob']

def test_spark_select_filter_by():
    # TODO: work-around issue with filter/select for now
    #t2 = t['name'][t['name'] != 'Alice']
    t2 = t[t['name'] != 'Alice']['name']
    t3 = t2.map(lambda x: x.lower(), schema='{name:string}')
    gby = By(t3, t3['name'], t3['name'].count())
    ansrdd = compute(gby, rdd)
    assert ansrdd.collect() == [('bob', 1)]
