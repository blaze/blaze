from __future__ import absolute_import

import sys
import string
from contextlib import contextmanager
from ..py3help import Counter

#------------------------------------------------------------------------
# Naming
#------------------------------------------------------------------------

_stream = None
_count = None
_depth = 0

def letters(source=string.ascii_lowercase):
    k = 0
    while 1:
        for a in source:
            yield a+str(k) if k else a
        k = k+1

@contextmanager
def namesupply():
    global _stream, _count, _depth
    _depth += 1
    if _depth == 1:
        _stream = letters()
        _count = Counter()
    yield _stream
    if _depth == 1:
        _stream = None
        _count = None
    _depth -= 1

def fresh(prefix=None):
    if _stream is None:
        raise Exception("Run with ``with namesupply():`` context.")

    if prefix:
        var = '%s%i' % (prefix, _count[prefix])
        _count[prefix] += 1
        return var
    else:
        return next(_stream)
