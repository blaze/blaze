from __future__ import absolute_import, division, print_function

__all__ = ['split_array_base', 'add_indexers_to_url', 'slice_as_string',
           'index_tuple_as_string']

from pyparsing import (Word, Regex, Optional, ZeroOrMore,
                       StringStart, StringEnd, alphas, alphanums)
from ..py2help import _strtypes, _inttypes

# Parser to match the Blaze URL syntax
intNumber = Regex(r'[-+]?\b\d+\b')
arrayName = Regex(r'(/(\.session_)?\w*)*[a-zA-Z0-9_]+\b')
bracketsIndexer = (Optional(intNumber) +
                   Optional(':' + Optional(intNumber)) +
                   Optional(':' + Optional(intNumber)))
indexerPattern = (('.' + Word(alphas + '_', alphanums + '_')) ^
                  ('[' + bracketsIndexer +
                   ZeroOrMore(',' + bracketsIndexer) + ']'))
arrayBase = (StringStart() +
             arrayName + ZeroOrMore(indexerPattern) +
             StringEnd())


def split_array_base(array_base):
    pieces = arrayBase.parseString(array_base)
    array_name = pieces[0]
    indexers = []
    i = 1
    while i < len(pieces):
        # Convert [...] into an int, a slice, or a tuple of int/slice
        if pieces[i] == '[':
            i += 1
            ilist = []
            while pieces[i-1] != ']':
                if pieces[i] != ':':
                    first = int(pieces[i])
                    i += 1
                else:
                    first = None
                if pieces[i] in [',', ']']:
                    i += 1
                    ilist.append(first)
                else:
                    i += 1
                    if pieces[i] not in [',', ':', ']']:
                        second = int(pieces[i])
                        i += 1
                    else:
                        second = None
                    if pieces[i] in [',', ']']:
                        i += 1
                        ilist.append(slice(first, second))
                    else:
                        i += 1
                        if pieces[i] not in [',', ']']:
                            third = int(pieces[i])
                            i += 1
                        else:
                            third = 1
                        ilist.append(slice(first, second, third))
                        i += 2
            if len(ilist) == 1:
                indexers.append(ilist[0])
            else:
                indexers.append(tuple(ilist))
        elif pieces[i] == '.':
            i += 1
        else:
            indexers.append(pieces[i])
            i += 1

    return array_name, indexers


def slice_as_interior_string(s):
    if type(s) is int:
        return str(s)
    else:
        result = ''
        if s.start is not None:
            result += str(s.start)
        result += ':'
        if s.stop is not None:
            result += str(s.stop)
        if s.step is not None and s.step != 1:
            result += ':' + str(s.step)
        return result


def slice_as_string(s):
    return '[' + slice_as_interior_string(s) + ']'


def index_tuple_as_string(s):
    result = '[' + slice_as_interior_string(s[0])
    for i in s[1:]:
        result += ',' + slice_as_interior_string(i)
    result += ']'
    return result


def add_indexers_to_url(base_url, indexers):
    for idx in indexers:
        if isinstance(idx, _strtypes):
            base_url += '.' + idx
        elif isinstance(idx, _inttypes):
            base_url += '[' + str(idx) + ']'
        elif isinstance(idx, slice):
            base_url += slice_as_string(idx)
        elif isinstance(idx, tuple):
            base_url += index_tuple_as_string(idx)
        else:
            raise IndexError('Cannot process index object %r' % idx)
    return base_url
