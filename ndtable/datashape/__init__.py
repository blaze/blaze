from parse import parse
from .coretypes import *

def dshape(s):
    return parse(s)

datashape = dshape
