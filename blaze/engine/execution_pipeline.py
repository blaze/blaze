"""
This module starts with an aterm graph and finds suitable executors for the
subgraphs that can be handled by that executor.
"""

import blaze
from blaze.engine import pipeline
from blaze.engine import llvm_execution

class ExecutionPipeline(object):

    def __init__(self):
        self.pipeline = [
            try_llvm,
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


def try_llvm(pipeline_context, aterm_graph):
    executors = pipeline_context['executors']
    llvm_execution.substitute_llvm_executors(executors)
