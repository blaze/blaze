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

t = TableSymbol('{name: string, amount: int, id: int}')

t2 = TableSymbol('{name: string, city: string}')

#Web Commons Graph Example data
data_idx = [['A', 1],
            ['B', 2],
            ['C', 3]]

data_arc = [[1, 3],
            [2, 3],
            [3, 1]]

t_idx = TableSymbol('{name: string, node_id: int32}')

t_arc = TableSymbol('{node_out: int32, node_id: int32}')

@skip("Spark not yet fully supported")
def test_table():
    assert compute(t, rdd) == rdd


@skip("Spark not yet fully supported")
def test_projection():
    assert compute(t['name'], rdd).collect() == rdd.map(lambda x:
                                                        x[0]).collect()


@skip("Spark not yet fully supported")
def test_multicols_projection():
    assert compute(t[['amount', 'name']], rdd).collect() == [[100, 'Alice'],
                                                             [200, 'Bob'],
                                                             [50, 'Alice']]


@skip("Spark not yet fully supported")
def test_selection():
    assert compute(t[t['name'] == 'Alice'], rdd).collect() ==\
        rdd.filter(lambda x: x[0] == 'Alice').collect()


@skip("Spark not yet fully supported")
def test_join():

    joined = Join(t, t2, 'name')
    expected = [['Alice', 100, 1, 'Austin'],
                ['Bob', 200, 2, 'Boston'],
                ['Alice', 50, 3, 'Austin']]
    result = compute(joined, rdd, rdd2)
    assert all([i in expected for i in result.collect()])

@skip("Spark not yet fully supported")
def test_groupby():

    rddidx = sc.parallelize(data_idx)
    rddarc = sc.parallelize(data_arc)

    joined = Join(t_arc, t_idx, "node_id")

    result_blaze = compute(joined, {t_arc: rddarc, t_idx:rddidx})
    t = By(joined, joined['name'], joined['node_id'].count())
    a = compute(t, {t_arc: rddarc, t_idx:rddidx})
    in_degree = dict(a.collect())
    assert in_degree['C'] == 2
    assert in_degree['A'] == 1

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





