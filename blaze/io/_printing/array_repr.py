from __future__ import absolute_import, division, print_function

from . import _arrayprint
from ...datadescriptor import RemoteDataDescriptor


def array_repr(a):
    # TODO: create a mechanism for data descriptor to override
    #       printing.
    if isinstance(a._data, RemoteDataDescriptor):
        body = 'RemoteDataDescriptor(\n%r)' % a._data.url
    else:
        body = _arrayprint.array2string(a._data, separator=', ')

    pre = 'array('
    post = ',\n' + ' '*len(pre) + "dshape='" + str(a.dshape) + "'" + ')'

    # For a multi-line, start it on the next line so things align properly
    if '\n' in body:
        pre += '\n'

    return pre + body + post
