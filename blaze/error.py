class error(Exception):
    "Base class for blaze exceptions"

class ExecutionError(error):
    """
    Raised when we are unable to execute a certain lazy or immediate
    expression.
    """