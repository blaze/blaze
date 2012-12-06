from blaze import *
from blaze.datashape import datashape
from blaze.engine import pipeline, llvm_execution


def convert_graph(lazy_blaze_graph):
    # Convert blaze graph to ATerm graph
    p = pipeline.Pipeline()
    context = p.run_pipeline_context(lazy_blaze_graph)
    aterm_graph = context['output']
    executors = {}
    aterm_graph = llvm_execution.substitute_llvm_executors(aterm_graph,
                                                           executors)
    return aterm_graph, executors

def test_conversion():
    a = NDArray([1, 2, 3, 4], datashape('2, 2, int32'))
    b = NDArray([5, 6, 7, 8], datashape('2, 2, float32'))
    c = NDArray([9, 10, 11, 12], datashape('2, 2, int32'))

    graph = a + b * c
    result_graph, executors = convert_graph(graph)

    assert result_graph.spine.label == 'Executor', result_graph.spine
    assert len(executors) == 1
    executor_id, executor = executors.popitem()
    assert str(result_graph.annotation) == "{numba,%d}" % executor_id

def test_execution():
    a = NDArray([1, 2, 3, 4], datashape('2, 2, float32'))
    b = NDArray([5, 6, 7, 8], datashape('2, 2, float32'))
    c = NDArray([9, 10, 11, 12], datashape('2, 2, float32'))

    graph = a + b * c
    a[:] = graph

    print a

if __name__ == '__main__':
#    test_conversion()
    test_execution()
