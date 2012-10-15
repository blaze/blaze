from functools import wraps
from nodes import UnaryOp, BinaryOp, NaryOp

def tablefunction(fn, nin, nout):
    """
    Wrap a arbitrary array function into a deferredk table function.
    """
    @wraps(fn)
    def wrapper(self):
        if nin == 1:
            return UnaryOp(fn, nin, nout)
        elif nin == 2:
            return BinaryOp(fn, nin, nout)
        elif nin > 2:
            return NaryOp(fn, nin, nout)

    return wrapper
