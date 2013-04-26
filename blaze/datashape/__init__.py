#import parser
from coretypes import *

def dopen(fname):
    contents = open(fname).read()
    return parser.parse_extern(contents)

def dshape(o):
    if isinstance(o, str):
        return parser.parse(o)
    elif isinstance(o, DataShape):
        return o
    else:
        raise TypeError('Cannot create dshape from object of type %s' % type(o))

datashape = dshape
