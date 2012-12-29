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

syntax_error = """

  File {filename}, line {lineno}
    {line}
    {pointer}

{error}: {msg}
"""

class CustomSyntaxError(Exception):
    """
    Makes datashape parse errors look like Python SyntaxError.
    """
    def __init__(self, lineno, col_offset, filename, text, msg=None):
        self.lineno     = lineno
        self.col_offset = col_offset
        self.filename   = filename
        self.text       = text
        self.msg        = msg or 'invalid syntax'
        raise NotImplementedError

    def __str__(self):
        return syntax_error.format(
            filename = self.filename,
            lineno   = self.lineno,
            line     = self.text,
            pointer  = ' '*self.col_offset + '^',
            msg      = self.msg,
            error    = self.__class__.__name__,
        )
