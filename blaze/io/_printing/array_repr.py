from __future__ import absolute_import, division, print_function

from . import _arrayprint
from ...datadescriptor import Remote_DDesc


def array_repr(a):
    pre = 'array('
    post = ',\n' + ' '*len(pre) + "dshape='" + str(a.dshape) + "'" + ')'

    # TODO: create a mechanism for data descriptor to override
    #       printing.
    if isinstance(a.ddesc, Remote_DDesc):
        body = 'Remote_DDesc(%r)' % a.ddesc.url
    else:
        body = _arrayprint.array2string(a.ddesc,
                          separator=', ',
                          prefix=' '*len(pre))

    return pre + body + post
