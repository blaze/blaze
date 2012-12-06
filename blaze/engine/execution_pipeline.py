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
            build_operand_dict,
            try_llvm,
            execute,
        ]

    def run_pipeline(self, pipeline_context, aterm_graph):
        # Map executor IDs to executor objects
        executors = {}
        pipeline_context['executors'] = executors

        for substitutor in self.pipeline:
            aterm_graph = substitutor(pipeline_context, aterm_graph)

        return pipeline_context['result']

def build_operand_dict(pipeline_context, aterm_graph):
    operands = pipeline_context['operands']
    operand_dict = dict((id(op), op) for op in operands)
    pipeline_context['operand_dict'] = operand_dict
    return aterm_graph

def try_llvm(pipeline_context, aterm_graph):
    "Substitute executors for the parts of the graph we can handle"
    executors = pipeline_context['executors']
    aterm_graph = llvm_execution.substitute_llvm_executors(aterm_graph, executors)
    return aterm_graph

def execute(pipeline_context, aterm_graph):
    "Execute the executor graph"
    operands = pipeline_context['operand_dict']
    executors = pipeline_context['executors']
    visitor = dispatch.ExecutorDispatcher(operands, executors)
    result = visitor.visit(aterm_graph)
    pipeline_context['result'] = result
    return aterm_graph