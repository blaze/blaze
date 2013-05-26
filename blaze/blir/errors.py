import sys
import logging
from contextlib import contextmanager

_count = 0
_errlog = []
_listeners = []

logger = logging.getLogger(__name__)

FORMAT = '%(message)s'
logging.basicConfig(format=FORMAT)
logging.basicConfig(level=logging.ERROR)

#------------------------------------------------------------------------
# Error Reporting
#------------------------------------------------------------------------

def error(lineno, message, filename=None):
    global _count, _errlog
    if not filename:
        errmsg = "{}: {}".format(lineno, message)
    else:
        errmsg = "{}:{}: {}".format(filename, lineno, message)
    for listener in _listeners:
        listener(errmsg)
    _errlog.append(errmsg)
    _count += 1

def _default_handler(msg):
    return logger.error(msg+"\n")

def reset():
    global _count
    global _errlog
    _count = 0

def log():
    global _errlog
    return _errlog

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
