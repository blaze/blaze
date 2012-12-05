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
    a = NDArray([1, 2, 3, 4], datashape('2, 2, int'))
    b = NDArray([5, 6, 7, 8], datashape('2, 2, int'))
    c = NDArray([9, 10, 11, 12], datashape('2, 2, int'))

    graph = a + b * c
    result_graph, executors = convert_graph(graph)
    print executors

test_conversion = None # disable

if __name__ == '__main__':
    test_conversion()