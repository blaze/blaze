import numpy as np

from blaze.expr import visitor, paterm
from blaze.engine import executors
from blaze.engine import llvm_execution


class ExecutorDispatcher(visitor.BasicGraphVisitor):
    """
    Evaluate an expression.

    TODO: execute each executor on a single chunk before moving to the next
          executor.
    """

    def __init__(self, arrays, executors):
        super(ExecutorDispatcher, self).__init__()
        self.arrays = arrays
        self.executors = executors

    def AAppl(self, node):
        """
        Executor(op1, op2, ..., opN, lhs_op?){'backend', executor_id, has_lhs}
        """
        # print node.annotation.meta, node.annotation

        if paterm.matches("Array;*", node.spine):
            id = node.annotation.meta[0]
            return self.arrays[id.label]
        else:
            assert  paterm.matches("Executor;*", node.spine), node

            backend, executor_id, has_lhs = node.annotation.meta
            executor = self.executors[executor_id.label]
            operands = self.visit(node.args)
            if has_lhs:
                lhs_op = operands.pop()
            else:
                # FIXME: !
                raise NotImplementedError

            executors.execute(executor, operands, lhs_op)
            return lhs_op
