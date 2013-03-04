import sys
from contextlib import contextmanager

_listeners = []
_count = 0

#------------------------------------------------------------------------
# Error Reporting
#------------------------------------------------------------------------

def error(lineno, message, filename=None):
    global _count
    if not filename:
        errmsg = "{}: {}".format(lineno, message)
    else:
        errmsg = "{}:{}: {}".format(filename, lineno, message)
    for listener in _listeners:
        listener(errmsg)
    _count += 1

def _default_handler(msg):
    sys.stderr.write(msg+"\n")
    return None

def reset():
    global _count
    _count = 0

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

def occurred():
    global _count
    return _count > 0

def reported():
    global _count
    return _count

@contextmanager
def listen(handler=_default_handler):
    _listeners.append(handler)
    try:
        yield
    finally:
        _listeners.remove(handler)
