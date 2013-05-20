__all__ = [
    'ExecutionError',
    'NoDispatch',
    'InvalidLibraryDefinition',
    'CustomSyntaxError',
    'StreamingDimensionError',
    'BroadcastError',
    'ArrayWriteError'
]

# TODO: BlazeException base class ?

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
class InvalidLibraryDefinition(Exception):
    pass

#------------------------------------------------------------------------
# Generic Syntax Errors
#------------------------------------------------------------------------

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
    def __init__(self, lexpos, filename, text, msg=None):
        self.lexpos   = lexpos
        self.filename = filename
        self.text     = text
        self.msg      = msg or 'invalid syntax'
        self.lineno = text.count('\n', 0, lexpos) + 1
        # Get the extent of the line with the error
        linestart = text.rfind('\n', 0, lexpos)
        if linestart < 0:
            linestart = 0
        else:
            linestart = linestart + 1
        lineend = text.find('\n', lexpos)
        if lineend < 0:
            lineend = len(text)
        self.line = text[linestart:lineend]
        self.col_offset = lexpos - linestart

    def __str__(self):
        pointer = ' '*self.col_offset + '^'

        return syntax_error.format(
            filename = self.filename,
            lineno   = self.lineno,
            line     = self.line,
            pointer  = pointer,
            msg      = self.msg,
            error    = self.__class__.__name__,
        )

    def __repr__(self):
        return str(self)

#------------------------------------------------------------------------
# Array-related errors
#------------------------------------------------------------------------

class StreamingDimensionError(Exception):
    """
    An error for when a streaming dimension is accessed
    like a dimension of known size.
    """
    pass

class BroadcastError(Exception):
    """
    An error for when arrays can't be broadcast together.
    """
    pass

class ArrayWriteError(Exception):
    """
    An error for when trying to write to an array which is read only.
    """
    pass
