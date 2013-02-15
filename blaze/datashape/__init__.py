import parser
from coretypes import *

def dshape(o):
    if isinstance(o, str):
        return parser.parse(o)
    elif isinstance(o, DataShape):
        return o

datashape = dshape
