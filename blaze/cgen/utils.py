import string
from copy import copy
from contextlib import contextmanager
from collections import Counter

#------------------------------------------------------------------------
# Naming
#------------------------------------------------------------------------

_stream = None
_count = None

def letters(source=string.ascii_lowercase):
    k = 0
    while 1:
        for a in source:
            yield a+str(k) if k else a
        k = k+1

@contextmanager
def namesupply():
    global _stream, _count
    _stream = letters()
    _count = Counter()
    yield _stream
    _stream = None
    _count = None

def anon(prefix=None):
    if _stream is None:
        raise Exception("Run with ``with names():`` context.")

    if prefix:
        var = '%s%i' % (prefix, _count[prefix])
        _count[prefix] += 1
        return var
    else:
        return next(_stream)
