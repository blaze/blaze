import parser
from coretypes import *

def dopen(fname):
    contents = open(fname).read()
    return parser.parse_extern(contents)

def dshape(o):
    if isinstance(o, str):
        return parser.parse(o)
    elif isinstance(o, DataShape):
        return o

datashape = dshape
