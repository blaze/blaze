import numpy as np

from blaze.expr import visitor
from blaze.engine import executors
from blaze.engine import llvm_execution


class ExecutorDispatcher(visitor.BasicGraphVisitor):
    """
    Evaluate an expression.

    TODO: execute each executor on a single chunk before moving to the next
          executor.
    """

    def __init__(self, executors):
        super(ExecutorDispatcher, self).__init__()
        self.executors = executors

    def Executor(self, node):
        """
        Executor(op1, op2, ..., opN, lhs_op?){'backend', executor_id, has_lhs}
        """
        backend, executor_id, has_lhs = node.annotation.meta
        executor = self.executors[executor_id]
        operands = self.visitchildren(node)
        if has_lhs:
            lhs_op = operands.pop()
        else:
            # FIXME: !
            raise NotImplementedError

        executors.execute(executor, operands, lhs_op)