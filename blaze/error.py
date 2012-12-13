class ChunkingException(BaseException):
    pass

class NoSuchChecksum(ValueError):
    pass

class ChecksumMismatch(RuntimeError):
    pass

class FileNotFound(IOError):
    pass

# for Numba
class ExecutionError(Exception):
    """
    Raised when we are unable to execute a certain lazy or immediate
    expression.
    """

# for the RTS
class NoDispatch(Exception):
    def __init__(self, aterm):
        self.aterm = aterm
    def __str__(self):
        return "No implementation for '%r'" % self.aterm

# for the RTS
class InvalidLibraryDefinton(Exception):
    pass

class NotNumpyCompatible(Exception):
    """
    Raised when we try to convert a datashape into a NumPy dtype
    but it cannot be ceorced.
    """
    pass
