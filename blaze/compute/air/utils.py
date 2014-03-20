from __future__ import print_function, division, absolute_import

try:
    import __builtin__ as builtins
except ImportError:
    import builtins

import string
import functools
import collections
from itertools import chain

map    = lambda *args: list(builtins.map(*args))
invert = lambda d: dict((v, k) for k, v in d.items())

def linearize(func):
    """
    Return a linearized from of the IR and a dict mapping basic blocks to
    offsets.
    """
    result = []
    blockstarts = {} # { block_label : instruction offset }
    for block in func.blocks:
        blockstarts[block.name] = len(result)
        result.extend(iter(block))

    return result, blockstarts

def nestedmap(f, args, type=list):
    """
    Map `f` over `args`, which contains elements or nested lists
    """
    result = []
    for arg in args:
        if isinstance(arg, type):
            result.append(list(map(f, arg)))
        else:
            result.append(f(arg))

    return result

def flatten(args):
    """Flatten nested lists (return as iterator)"""
    for arg in args:
        if isinstance(arg, list):
            for x in arg:
                yield x
        else:
            yield arg

def mutable_flatten(args):
    """Flatten nested lists (return as iterator)"""
    for arg in args:
        if isinstance(arg, list):
            for x in arg:
                yield x
        else:
            yield arg

def mergedicts(*dicts):
    """Merge all dicts into a new dict"""
    return dict(chain(*[d.items() for d in dicts]))

def listify(f):
    """Decorator to turn generator results into lists"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return list(f(*args, **kwargs))
    return wrapper

@listify
def listitems(fields):
    """Turn [1, [2, 3], (4,)] into [[1], [2, 3], [4]]"""
    for x in fields:
        if not isinstance(x, (list, tuple)):
            yield [x]
        else:
            yield list(x)

@listify
def prefix(iterable, prefix):
    """Prefix each item from the iterable with a prefix"""
    for item in iterable:
        yield prefix + item

# ______________________________________________________________________
# Strings

def substitute(s, **substitutions):
    """Use string.Template to substitute placeholders in a string"""
    return string.Template(s).substitute(**substitutions)

# ______________________________________________________________________

def hashable(x):
    try:
        hash(x)
    except TypeError:
        return False
    else:
        return True

# ______________________________________________________________________

class ValueDict(object):
    """
    Use dict values as attributes.
    """

    def __init__(self, d):
        self.__getattr__ = d.__getitem__
        self.__setattr__ = d.__setitem__
        self.__detattr__ = d.__detitem__

# ______________________________________________________________________

def call_once(f):
    """Cache the result of the function, so that it's called only once"""
    result = []
    def wrapper(*args, **kwargs):
        if len(result) == 0:
            ret = f(*args, **kwargs)
            result.append(ret)

        return result[0]
    return wrapper

def cached(limit=1000):
    """Cache the result for the arguments just once"""
    def decorator(f):
        cache = {}
        def wrapper(*args):
            if args not in cache:
                if len(cache) > limit:
                    cache.popitem()
                cache[args] = f(*args)
            return cache[args]
        return wrapper
    return decorator

# ______________________________________________________________________

def make_temper():
    """Return a function that returns temporary names"""
    temps = collections.defaultdict(int)
    seen = set()

    def temper(input=""):
        name, dot, tail = input.rpartition('.')
        if tail.isdigit():
            varname = name
        else:
            varname = input

        count = temps[varname]
        temps[varname] += 1
        if varname and count == 0:
            result = varname
        else:
            result = "%s.%d" % (varname, count)

        assert result not in seen
        seen.add(result)

        return result

    return temper

# ______________________________________________________________________


def _getops(func_or_block_or_list):
    if isinstance(func_or_block_or_list, list):
        return func_or_block_or_list
    return func_or_block_or_list.ops


def findop(container, opcode):
    """Find the first Operation with the given opcode"""
    for op in _getops(container):
        if op.opcode == opcode:
            return op


def findallops(container, opcode):
    """Find all Operations with the given opcode"""
    found = []
    for op in _getops(container):
        if op.opcode == opcode:
            found.append(op)

    return found
