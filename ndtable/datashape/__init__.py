from parse import parse
from .coretypes import *

def dshape(o):
    if isinstance(o, str):
        return parse(o)
    elif isinstance(s, DataShape):
        return o

datashape = dshape
