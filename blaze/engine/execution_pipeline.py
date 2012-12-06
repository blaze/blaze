"""
This module starts with an aterm graph and finds suitable executors for the
subgraphs that can be handled by that executor.
"""

import blaze

from blaze.engine import pipeline
from blaze.engine import llvm_execution
from blaze.engine import dispatch

class ExecutionPipeline(object):

    def __init__(self):
        self.pipeline = [
            try_llvm,
            execute,
        ]

    def run_pipeline(self, pipeline_context, aterm_graph):
        # Map executor IDs to executor objects
        executors = {}
        pipeline_context['executors'] = executors

        for substitutor in self.pipeline:
            aterm_graph = substitutor(pipeline_context, aterm_graph)

        # TODO: Match recursively to see whether we handled the entire graph
        full_substitution = aterm_graph.matches('Executor;*')
        if not full_substitution:
            # TODO: find and pretty-print faulty sub-expression
            raise blaze.ExecutionError("Unable to execute (sub-)expression")

        return pipeline_context['result']

def try_llvm(pipeline_context, aterm_graph):
    "Substitute executors for the parts of the graph we can handle"
    executors = pipeline_context['executors']
    aterm_graph = llvm_execution.substitute_llvm_executors(aterm_graph, executors)
    return aterm_graph

def execute(pipeline_context, aterm_graph):
    "Execute the executor graph"
    executors = pipeline_context['executors']
    visitor = dispatch.ExecutorDispatcher(executors)
    result = visitor.visit(aterm_graph)
    pipeline_context['result'] = result
    return aterm_graph