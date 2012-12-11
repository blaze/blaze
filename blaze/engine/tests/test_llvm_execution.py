from blaze import *
from blaze.datashape import datashape
from blaze.engine import pipeline, llvm_execution

from unittest import skip

def convert_graph(lazy_blaze_graph):
    # Convert blaze graph to ATerm graph
    p = pipeline.Pipeline(have_numbapro=True)
    context, aterm_graph = p.run_pipeline(lazy_blaze_graph)

    executors = {}
    aterm_graph = llvm_execution.substitute_llvm_executors(aterm_graph,
                                                           executors)
    return aterm_graph, executors

@skip("Unstable")
def test_conversion():
    a = NDArray([1, 2, 3, 4], datashape('2, 2, int32'))
    b = NDArray([5, 6, 7, 8], datashape('2, 2, float32'))
    c = NDArray([9, 10, 11, 12], datashape('2, 2, int32'))

    graph = a + b * c
    result_graph, executors = convert_graph(graph)

    assert result_graph.spine.label == 'Executor', result_graph.spine
    assert len(executors) == 1
    executor_id, executor = executors.popitem()
    #assert str(result_graph.annotation) == "{numba,%d}" % executor_id

@skip("Unstable")
def test_execution():
    a = NDArray([1, 2, 3, 4], datashape('4, float32'))
    b = NDArray([5, 6, 7, 8], datashape('4, float32'))
    c = NDArray([9, 10, 11, 12], datashape('4, float32'))
    out = NDArray([0, 0, 0, 0], datashape('4, float32'))

    graph = a + b * c
    out[:] = graph

    print list(out.data.ca)
    assert list(out.data.ca) == [46, 62, 80, 100]

if __name__ == '__main__':
    # XXX: huh, if I run these both it seems to segfault
#   test_conversion()
   test_execution()

