from blaze.expr import visitor
from blaze.engine import executors

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
        backend, executor_id = node.annotation.meta
        executor = self.executors[executor_id]
        operands = self.visitchildren(node)
        executors.execute(executor, operands)