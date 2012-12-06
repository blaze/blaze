class error(Exception):
    "Base class for blaze exceptions"

class ExecutionError(error):
    """
    Raised when we are unable to execute a certain lazy or immediate
    expression.
    """

class NotNumpyCompatible(Exception):
    """
    Raised when we try to convert a datashape into a NumPy dtype
    but it cannot be ceorced.
    """
    pass
