class CompileError(Exception):
    """Raised for various sorts of compilation errors"""

class PositioningError(CompileError):
    """Raised when a builder cannot be positioned as requested"""

class IRError(CompileError):
    """Raised when the IR is somehow malformed"""