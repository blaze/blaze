from __future__ import absolute_import

from . import _arrayprint

def array_repr(a):
    pre = 'array('
    post =  ',\n' + ' '*len(pre) + "dshape='" + str(a.dshape) + "'" + ')'
    body = _arrayprint.array2string(a._data,
                      separator=', ',
                      prefix=' '*len(pre))

    return pre + body + post
