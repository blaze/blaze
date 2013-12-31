from __future__ import absolute_import, division, print_function

from . import _arrayprint


def array_repr(a):
    body = _arrayprint.array2string(a._data, separator=', ')
    pre = 'array('
    post = ',\n' + ' '*len(pre) + "dshape='" + str(a.dshape) + "'" + ')'

    # For a multi-line, start it on the next line so things align properly
    if '\n' in body:
        pre += '\n'

    return pre + body + post
